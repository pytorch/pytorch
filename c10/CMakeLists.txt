project(c10 CXX)

include(CheckCXXSourceCompiles)

check_cxx_source_compiles("#include <execinfo.h>
int main() { return 0; }" C10_HAS_EXECINFO)
check_cxx_source_compiles("#include <cxxabi.h>
int main() { return 0; }" C10_HAS_CXXABI)

configure_file(Config.h.in c10/Config.h)

file(GLOB C10_SRCS *.cpp)
# NB: ATen's install rule assumes that these headers are flat.  If this is not
# true you'll have to fix this
file(GLOB C10_HEADERS *.h)
set(C10_SRCS ${C10_SRCS} PARENT_SCOPE)
set(C10_HEADERS "${C10_HEADERS}" "${CMAKE_CURRENT_BINARY_DIR}/c10/Config.h" PARENT_SCOPE)
set(C10_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
