# Bruce C Compiler ignores "-g" flag and optimization cannot be
# enabled here (it is implemented only for 8086 target).
string(APPEND CMAKE_C_FLAGS_INIT " -D__CLASSIC_C__")
string(APPEND CMAKE_C_FLAGS_DEBUG_INIT " -g")
string(APPEND CMAKE_C_FLAGS_MINSIZEREL_INIT " -DNDEBUG")
string(APPEND CMAKE_C_FLAGS_RELEASE_INIT " -DNDEBUG")
string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO_INIT " -g -DNDEBUG")

set(CMAKE_C_LINKER_WRAPPER_FLAG "-X")

set(CMAKE_C_LINK_MODE DRIVER)
