include(Platform/AIX-XL)
__aix_compiler_xl(C)

# -qhalt=e       = Halt on error messages (rather than just severe errors)
string(APPEND CMAKE_C_FLAGS_INIT " -qhalt=e")
