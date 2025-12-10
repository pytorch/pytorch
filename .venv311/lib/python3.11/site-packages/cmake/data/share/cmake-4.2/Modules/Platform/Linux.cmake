set(CMAKE_DL_LIBS "dl")
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-rpath,")
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP ":")
set(CMAKE_SHARED_LIBRARY_RPATH_ORIGIN_TOKEN "\$ORIGIN")
set(CMAKE_SHARED_LIBRARY_RPATH_LINK_C_FLAG "-Wl,-rpath-link,")
set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Wl,-soname,")
set(CMAKE_EXE_EXPORTS_C_FLAG "-Wl,--export-dynamic")

# Shared libraries with no builtin soname may not be linked safely by
# specifying the file path.
set(CMAKE_PLATFORM_USES_PATH_WHEN_NO_SONAME 1)

# Initialize C link type selection flags.  These flags are used when
# building a shared library, shared module, or executable that links
# to other libraries to select whether to use the static or shared
# versions of the libraries.
foreach(type SHARED_LIBRARY SHARED_MODULE EXE)
  set(CMAKE_${type}_LINK_STATIC_C_FLAGS "-Wl,-Bstatic")
  set(CMAKE_${type}_LINK_DYNAMIC_C_FLAGS "-Wl,-Bdynamic")
endforeach()


# Features for LINK_GROUP generator expression
## RESCAN: request the linker to rescan static libraries until there is
## no pending undefined symbols
set(CMAKE_LINK_GROUP_USING_RESCAN "LINKER:--start-group" "LINKER:--end-group")
set(CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED TRUE)


# Debian policy requires that shared libraries be installed without
# executable permission.  Fedora policy requires that shared libraries
# be installed with the executable permission.  Since the native tools
# create shared libraries with execute permission in the first place a
# reasonable policy seems to be to install with execute permission by
# default.  In order to support debian packages we provide an option
# here.  The option default is based on the current distribution, but
# packagers can set it explicitly on the command line.
if(DEFINED CMAKE_INSTALL_SO_NO_EXE)
  # Store the decision variable in the cache.  This preserves any
  # setting the user provides on the command line.
  set(CMAKE_INSTALL_SO_NO_EXE "${CMAKE_INSTALL_SO_NO_EXE}" CACHE INTERNAL
    "Install .so files without execute permission.")
else()
  # Store the decision variable as an internal cache entry to avoid
  # checking the platform every time.  This option is advanced enough
  # that only package maintainers should need to adjust it.  They are
  # capable of providing a setting on the command line.
  if(EXISTS "/etc/debian_version")
    set(CMAKE_INSTALL_SO_NO_EXE 1 CACHE INTERNAL
      "Install .so files without execute permission.")
  else()
    set(CMAKE_INSTALL_SO_NO_EXE 0 CACHE INTERNAL
      "Install .so files without execute permission.")
  endif()
endif()

include(Platform/UnixPaths)

# Debian has lib32 and lib64 paths only for compatibility so they should not be
# searched.
if(NOT CMAKE_CROSSCOMPILING AND NOT CMAKE_COMPILER_SYSROOT)
  if (EXISTS "/etc/debian_version")
    set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB32_PATHS FALSE)
    set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE)
  endif()
  if (EXISTS "/etc/arch-release")
    set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE)
  endif()
endif()
