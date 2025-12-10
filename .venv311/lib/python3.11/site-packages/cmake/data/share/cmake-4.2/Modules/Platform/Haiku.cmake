set(CMAKE_DL_LIBS "")
set(CMAKE_SHARED_LIBRARY_C_FLAGS "-fPIC")
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-shared")
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-rpath,")
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP ":")
set(CMAKE_SHARED_LIBRARY_RPATH_ORIGIN_TOKEN "\$ORIGIN")
set(CMAKE_SHARED_LIBRARY_RPATH_LINK_C_FLAG "-Wl,-rpath-link,")
set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Wl,-soname,")
set(CMAKE_EXE_EXPORTS_C_FLAG "-Wl,--export-dynamic")

# Determine, if the C or C++ compiler is configured for a secondary
# architecture. If so, that will change the search paths we set below. We check
# whether the compiler's library search paths contain a
# "/boot/system/develop/lib/<subdir>/", which we assume to be the secondary
# architecture specific subdirectory and extract the name of the architecture
# accordingly.

# First of all, find a C or C++ compiler we can run. The "arg1" is necessary
# here for compilers such as "distcc gcc-x86" or "ccache gcc-x86"
# TODO See CMakeDetermineCompilerId.cmake for some more things we may want to do.
if(CMAKE_C_COMPILER)
  set(__HAIKU_COMPILER ${CMAKE_C_COMPILER})
  string (STRIP "${CMAKE_C_COMPILER_ARG1}" __HAIKU_COMPILER_FLAGS)
else()
  set(__HAIKU_COMPILER ${CMAKE_CXX_COMPILER})
  string (STRIP "${CMAKE_CXX_COMPILER_ARG1}" __HAIKU_COMPILER_FLAGS)
endif()


execute_process(
  COMMAND ${__HAIKU_COMPILER} ${__HAIKU_COMPILER_FLAGS} -print-search-dirs
  OUTPUT_VARIABLE _HAIKU_SEARCH_DIRS
  RESULT_VARIABLE _HAIKU_SEARCH_DIRS_FOUND
  OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX MATCH "libraries: =?([^\n]*:)?/boot/system/develop/lib/([^/]*)/?(:?\n+)" _dummy "${_HAIKU_SEARCH_DIRS}\n")
set(CMAKE_HAIKU_SECONDARY_ARCH "${CMAKE_MATCH_2}")

if(NOT CMAKE_HAIKU_SECONDARY_ARCH)
  set(CMAKE_HAIKU_SECONDARY_ARCH_SUBDIR "")
  unset(CMAKE_HAIKU_SECONDARY_ARCH)
else()
  set(CMAKE_HAIKU_SECONDARY_ARCH_SUBDIR "/${CMAKE_HAIKU_SECONDARY_ARCH}")

  # Override CMAKE_*LIBRARY_ARCHITECTURE. This will cause FIND_LIBRARY to search
  # the libraries in the correct subdirectory first. It still isn't completely
  # correct, since the parent directories shouldn't be searched at all. The
  # primary architecture library might still be found, if there isn't one
  # installed for the secondary architecture or it is installed in a less
  # specific location.
  set(CMAKE_LIBRARY_ARCHITECTURE ${CMAKE_HAIKU_SECONDARY_ARCH})
  set(CMAKE_C_LIBRARY_ARCHITECTURE ${CMAKE_HAIKU_SECONDARY_ARCH})
  set(CMAKE_CXX_LIBRARY_ARCHITECTURE ${CMAKE_HAIKU_SECONDARY_ARCH})
endif()

list(APPEND CMAKE_SYSTEM_PREFIX_PATH
  /boot/system/non-packaged
  /boot/system
  )

list(APPEND CMAKE_HAIKU_COMMON_INCLUDE_DIRECTORIES
  /boot/system/non-packaged/develop/headers${CMAKE_HAIKU_SECONDARY_ARCH_SUBDIR}
  /boot/system/develop/headers/os
  /boot/system/develop/headers/os/app
  /boot/system/develop/headers/os/device
  /boot/system/develop/headers/os/drivers
  /boot/system/develop/headers/os/game
  /boot/system/develop/headers/os/interface
  /boot/system/develop/headers/os/kernel
  /boot/system/develop/headers/os/locale
  /boot/system/develop/headers/os/mail
  /boot/system/develop/headers/os/media
  /boot/system/develop/headers/os/midi
  /boot/system/develop/headers/os/midi2
  /boot/system/develop/headers/os/net
  /boot/system/develop/headers/os/opengl
  /boot/system/develop/headers/os/storage
  /boot/system/develop/headers/os/support
  /boot/system/develop/headers/os/translation
  /boot/system/develop/headers/os/add-ons/graphics
  /boot/system/develop/headers/os/add-ons/input_server
  /boot/system/develop/headers/os/add-ons/screen_saver
  /boot/system/develop/headers/os/add-ons/tracker
  /boot/system/develop/headers/os/be_apps/Deskbar
  /boot/system/develop/headers/os/be_apps/NetPositive
  /boot/system/develop/headers/os/be_apps/Tracker
  /boot/system/develop/headers/3rdparty
  /boot/system/develop/headers/bsd
  /boot/system/develop/headers/glibc
  /boot/system/develop/headers/gnu
  /boot/system/develop/headers/posix
  /boot/system/develop/headers${CMAKE_HAIKU_SECONDARY_ARCH_SUBDIR}
  )
if(CMAKE_HAIKU_SECONDARY_ARCH)
  list(APPEND CMAKE_HAIKU_COMMON_INCLUDE_DIRECTORIES
    /boot/system/develop/headers
    )
endif()

list(APPEND CMAKE_HAIKU_C_INCLUDE_DIRECTORIES
  ${CMAKE_HAIKU_COMMON_INCLUDE_DIRECTORIES}
  )

list(APPEND CMAKE_HAIKU_CXX_INCLUDE_DIRECTORIES
  ${CMAKE_HAIKU_COMMON_INCLUDE_DIRECTORIES})

list(APPEND CMAKE_SYSTEM_INCLUDE_PATH ${CMAKE_HAIKU_C_INCLUDE_DIRECTORIES})

list(APPEND CMAKE_HAIKU_DEVELOP_LIB_DIRECTORIES
  /boot/system/non-packaged/develop/lib${CMAKE_HAIKU_SECONDARY_ARCH_SUBDIR}
  /boot/system/develop/lib${CMAKE_HAIKU_SECONDARY_ARCH_SUBDIR}
  )

list(APPEND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
  ${CMAKE_HAIKU_DEVELOP_LIB_DIRECTORIES}
  )

list(APPEND CMAKE_SYSTEM_LIBRARY_PATH ${CMAKE_HAIKU_DEVELOP_LIB_DIRECTORIES})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/boot/system" CACHE PATH
    "Install path prefix, prepended onto install directories." FORCE)
endif()
