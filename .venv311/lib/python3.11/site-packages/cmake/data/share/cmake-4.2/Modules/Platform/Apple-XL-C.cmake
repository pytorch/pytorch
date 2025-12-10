set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS "-bundle")

# Enable shared library versioning.
set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Wl,-install_name")

# -qhalt=e       = Halt on error messages (rather than just severe errors)
string(APPEND CMAKE_C_FLAGS_INIT " -qhalt=e")
