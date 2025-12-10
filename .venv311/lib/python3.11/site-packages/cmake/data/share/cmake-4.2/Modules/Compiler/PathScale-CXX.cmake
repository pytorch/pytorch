include(Compiler/PathScale)
__compiler_pathscale(CXX)
string(APPEND CMAKE_CXX_FLAGS_MINSIZEREL_INIT " -DNDEBUG")
string(APPEND CMAKE_CXX_FLAGS_RELEASE_INIT " -DNDEBUG")
