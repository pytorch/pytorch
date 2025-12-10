set(CMAKE_SHARED_LIBRARY_PREFIX "lib")          # lib
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")          # .so
set(CMAKE_SHARED_LIBRARY_ARCHIVE_SUFFIX ".a")   # .a
set(CMAKE_AIX_IMPORT_FILE_PREFIX "")
set(CMAKE_AIX_IMPORT_FILE_SUFFIX ".imp")
set(CMAKE_DL_LIBS "ld")

# RPATH support on AIX is called libpath.  By default the runtime
# libpath is paths specified by -L followed by /usr/lib and /lib.  In
# order to prevent the -L paths from being used we must force use of
# -Wl,-blibpath:/usr/lib:/lib whether RPATH support is on or not.
# When our own RPATH is to be added it may be inserted before the
# "always" paths.
if(NOT DEFINED CMAKE_PLATFORM_REQUIRED_RUNTIME_PATH)
  set(CMAKE_PLATFORM_REQUIRED_RUNTIME_PATH /usr/lib /lib)
endif()

# Files named "libfoo.a" may actually be shared libraries.
set_property(GLOBAL PROPERTY TARGET_ARCHIVES_MAY_BE_SHARED_LIBS 1)

# since .a can be a static or shared library on AIX, we can not do this.
# at some point if we wanted it, we would have to figure out if a .a is
# static or shared, then we could add this back:

# Initialize C link type selection flags.  These flags are used when
# building a shared library, shared module, or executable that links
# to other libraries to select whether to use the static or shared
# versions of the libraries.
#foreach(type SHARED_LIBRARY SHARED_MODULE EXE)
#  set(CMAKE_${type}_LINK_STATIC_C_FLAGS "-bstatic")
#  set(CMAKE_${type}_LINK_DYNAMIC_C_FLAGS "-bdynamic")
#endforeach()

include(Platform/UnixPaths)
