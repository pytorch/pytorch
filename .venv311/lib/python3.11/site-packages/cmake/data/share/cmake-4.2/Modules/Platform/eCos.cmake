# support for eCos http://ecos.sourceware.org

# Guard against multiple inclusion, which e.g. leads to multiple calls to add_definition() #12987
if(__ECOS_CMAKE_INCLUDED)
  return()
endif()
set(__ECOS_CMAKE_INCLUDED TRUE)

set(CMAKE_SHARED_LIBRARY_C_FLAGS "")            # -pic
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "")       # -shared
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")         # +s, flag for exe link to use shared lib
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "")       # -rpath
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP "")   # : or empty

set(CMAKE_LINK_LIBRARY_SUFFIX "")
set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")          # lib
set(CMAKE_SHARED_LIBRARY_SUFFIX ".a")           # .a
set(CMAKE_EXECUTABLE_SUFFIX ".elf")             # same suffix as if built using UseEcos.cmake
set(CMAKE_DL_LIBS "" )

set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")


include(Platform/UnixPaths)

# eCos can be built only with gcc
get_property(_IN_TC GLOBAL PROPERTY IN_TRY_COMPILE)
if(CMAKE_C_COMPILER AND NOT  CMAKE_C_COMPILER_ID MATCHES "GNU" AND NOT _IN_TC)
  message(FATAL_ERROR "GNU gcc is required for eCos")
endif()
if(CMAKE_CXX_COMPILER AND NOT  "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" AND NOT _IN_TC)
  message(FATAL_ERROR "GNU g++ is required for eCos")
endif()

# find eCos system files
find_path(ECOS_SYSTEM_CONFIG_HEADER_PATH NAMES pkgconf/system.h)
find_library(ECOS_SYSTEM_TARGET_LIBRARY NAMES libtarget.a)

if(NOT ECOS_SYSTEM_CONFIG_HEADER_PATH)
  message(FATAL_ERROR "Could not find eCos pkgconf/system.h. Build eCos first and set up CMAKE_FIND_ROOT_PATH correctly.")
endif()

if(NOT ECOS_SYSTEM_TARGET_LIBRARY)
  message(FATAL_ERROR "Could not find eCos \"libtarget.a\". Build eCos first and set up CMAKE_FIND_ROOT_PATH correctly.")
endif()

get_filename_component(ECOS_LIBTARGET_DIRECTORY "${ECOS_SYSTEM_TARGET_LIBRARY}" PATH)
include_directories(${ECOS_SYSTEM_CONFIG_HEADER_PATH})
add_definitions(-D__ECOS__=1 -D__ECOS=1)

# special link commands for eCos executables
set(CMAKE_CXX_LINK_EXECUTABLE  "<CMAKE_CXX_COMPILER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> -nostdlib -nostartfiles -L${ECOS_LIBTARGET_DIRECTORY} -Ttarget.ld <LINK_LIBRARIES>")
set(CMAKE_C_LINK_EXECUTABLE    "<CMAKE_C_COMPILER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> -nostdlib -nostartfiles -L${ECOS_LIBTARGET_DIRECTORY} -Ttarget.ld <LINK_LIBRARIES>")

# eCos doesn't support shared libs
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)

set(CMAKE_CXX_LINK_SHARED_LIBRARY )
set(CMAKE_CXX_LINK_MODULE_LIBRARY )
set(CMAKE_C_LINK_SHARED_LIBRARY )
set(CMAKE_C_LINK_MODULE_LIBRARY )
