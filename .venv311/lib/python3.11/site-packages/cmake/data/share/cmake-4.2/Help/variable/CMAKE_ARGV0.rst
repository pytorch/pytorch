CMAKE_ARGV0
-----------

Command line argument passed to CMake in script mode.

When run in :ref:`-P <Script Processing Mode>` script mode, CMake sets this
variable to the first command line argument.  It then also sets ``CMAKE_ARGV1``,
``CMAKE_ARGV2``, ... and so on, up to the number of command line arguments
given.  See also :variable:`CMAKE_ARGC`.
