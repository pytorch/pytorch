set(CMAKE_ASM_VERBOSE_FLAG "-V")

# -qthreaded     = Ensures that all optimizations will be thread-safe
# -qhalt=e       = Halt on error messages (rather than just severe errors)
string(APPEND CMAKE_ASM_FLAGS_INIT " -qthreaded -qhalt=e -qsourcetype=assembler")

string(APPEND CMAKE_ASM_FLAGS_DEBUG_INIT " -g")
string(APPEND CMAKE_ASM_FLAGS_RELEASE_INIT " -O -DNDEBUG")
string(APPEND CMAKE_ASM_FLAGS_MINSIZEREL_INIT " -O -DNDEBUG")
string(APPEND CMAKE_ASM_FLAGS_RELWITHDEBINFO_INIT " -g -DNDEBUG")

set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s )
