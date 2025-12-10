CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE
--------------------------------------

Automatically add the current source and build directories to the
:prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` target property.

If this variable is enabled, CMake automatically adds for each shared
library target, static library target, module target and executable
target, :variable:`CMAKE_CURRENT_SOURCE_DIR` and
:variable:`CMAKE_CURRENT_BINARY_DIR` to
the :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` target property.  By default
``CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE`` is ``OFF``.
