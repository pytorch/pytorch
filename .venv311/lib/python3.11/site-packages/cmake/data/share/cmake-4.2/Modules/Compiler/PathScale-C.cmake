include(Compiler/PathScale)
__compiler_pathscale(C)
string(APPEND CMAKE_C_FLAGS_MINSIZEREL_INIT " -DNDEBUG")
string(APPEND CMAKE_C_FLAGS_RELEASE_INIT " -DNDEBUG")
