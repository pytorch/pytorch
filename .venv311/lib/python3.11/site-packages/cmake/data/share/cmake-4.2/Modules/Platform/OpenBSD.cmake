include(Platform/NetBSD)

# On OpenBSD, the compile time linker does not share it's configuration with
# the runtime linker.  This will extract the library search paths from the
# system's ld.so.hints file which will allow CMake to set the appropriate
# -rpath-link flags
if(NOT CMAKE_PLATFORM_RUNTIME_PATH)
  execute_process(COMMAND /sbin/ldconfig -r
                  OUTPUT_VARIABLE LDCONFIG_HINTS
                  ERROR_QUIET)
  string(REGEX REPLACE ".*search\\ directories:\\ ([^\n]*).*" "\\1"
         LDCONFIG_HINTS "${LDCONFIG_HINTS}")
  string(REPLACE ":" ";"
         CMAKE_PLATFORM_RUNTIME_PATH
         "${LDCONFIG_HINTS}")
endif()

# OpenBSD requires -z origin to enable $ORIGIN expansion in RPATH.
# This is not required for NetBSD.
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-z,origin,-rpath,")

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_OPENBSD_VERSIONING 1)

# OpenBSD has no multilib
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB32_PATHS FALSE)
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE)

# OpenBSD policy requires that shared libraries be installed without
# executable permission.
set(CMAKE_INSTALL_SO_NO_EXE 1)

if($ENV{LOCALBASE})
  set(OPENBSD_LOCALBASE $ENV{LOCALBASE})
else()
  set(OPENBSD_LOCALBASE /usr/local)
endif()
if($ENV{X11BASE})
  set(OPENBSD_X11BASE $ENV{X11BASE})
else()
  set(OPENBSD_X11BASE /usr/X11R6)
endif()

list(APPEND CMAKE_SYSTEM_PREFIX_PATH ${OPENBSD_LOCALBASE})
