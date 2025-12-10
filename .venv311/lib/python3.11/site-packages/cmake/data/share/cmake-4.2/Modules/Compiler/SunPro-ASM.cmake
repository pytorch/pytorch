include(Compiler/SunPro)
__compiler_sunpro(ASM)

set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s )

set(CMAKE_ASM_VERBOSE_FLAG "-#")

set(CMAKE_SHARED_LIBRARY_ASM_FLAGS "-KPIC")
set(CMAKE_SHARED_LIBRARY_CREATE_ASM_FLAGS "-G")
set(CMAKE_SHARED_LIBRARY_RUNTIME_ASM_FLAG "-R")
set(CMAKE_SHARED_LIBRARY_RUNTIME_ASM_FLAG_SEP ":")
set(CMAKE_SHARED_LIBRARY_SONAME_ASM_FLAG "-h")

string(APPEND CMAKE_ASM_FLAGS_INIT " ")
string(APPEND CMAKE_ASM_FLAGS_DEBUG_INIT " -g")
string(APPEND CMAKE_ASM_FLAGS_MINSIZEREL_INIT " -xO2 -xspace -DNDEBUG")
string(APPEND CMAKE_ASM_FLAGS_RELEASE_INIT " -xO3 -DNDEBUG")
string(APPEND CMAKE_ASM_FLAGS_RELWITHDEBINFO_INIT " -g -xO2 -DNDEBUG")

# Initialize ASM link type selection flags.  These flags are used when
# building a shared library, shared module, or executable that links
# to other libraries to select whether to use the static or shared
# versions of the libraries.
foreach(type SHARED_LIBRARY SHARED_MODULE EXE)
  set(CMAKE_${type}_LINK_STATIC_ASM_FLAGS "-Bstatic")
  set(CMAKE_${type}_LINK_DYNAMIC_ASM_FLAGS "-Bdynamic")
endforeach()
