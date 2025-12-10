CMAKE_HOST_SYSTEM_VERSION
-------------------------

The OS version CMake is running on.

A numeric version string for the system.  On systems that support
``uname``, this variable is set to the output of ``uname -r``. On other
systems this is set to major-minor version numbers.
