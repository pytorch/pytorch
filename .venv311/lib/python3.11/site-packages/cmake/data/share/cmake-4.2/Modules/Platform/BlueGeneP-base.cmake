# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


#
# BlueGeneP base platform file.
#
# NOTE: Do not set your platform to "BlueGeneP-base".  This file is included
# by the real platform files.  Use one of these two platforms instead:
#
#     BlueGeneP-dynamic  For dynamically linked builds
#     BlueGeneP-static   For statically linked builds
#
# This platform file tries its best to adhere to the behavior of the MPI
# compiler wrappers included with the latest BG/P drivers.
#


#
# For BGP builds, we're cross compiling, but we don't want to re-root things
# (e.g. with CMAKE_FIND_ROOT_PATH) because users may have libraries anywhere on
# the shared filesystems, and this may lie outside the root.  Instead, we set the
# system directories so that the various system BGP CNK library locations are
# searched first.  This is not the clearest thing in the world, given IBM's driver
# layout, but this should cover all the standard ones.
#
set(CMAKE_SYSTEM_LIBRARY_PATH
  /bgsys/drivers/ppcfloor/comm/default/lib                # default comm layer (used by mpi compiler wrappers)
  /bgsys/drivers/ppcfloor/comm/sys/lib                    # DCMF, other lower-level comm libraries
  /bgsys/drivers/ppcfloor/runtime/SPI                     # other low-level stuff
  /bgsys/drivers/ppcfloor/gnu-linux/lib                   # CNK python installation directory
  /bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib # CNK Linux image -- standard runtime libs, pthread, etc.
)

#
# This adds directories that find commands should specifically ignore for cross compiles.
# Most of these directories are the includeand lib directories for the frontend on BG/P systems.
# Not ignoring these can cause things like FindX11 to find a frontend PPC version mistakenly.
# We use this on BG instead of re-rooting because backend libraries are typically strewn about
# the filesystem, and we can't re-root ALL backend libraries to a single place.
#
set(CMAKE_SYSTEM_IGNORE_PATH
  /lib             /lib64             /include
  /usr/lib         /usr/lib64         /usr/include
  /usr/local/lib   /usr/local/lib64   /usr/local/include
  /usr/X11/lib     /usr/X11/lib64     /usr/X11/include
  /usr/lib/X11     /usr/lib64/X11     /usr/include/X11
  /usr/X11R6/lib   /usr/X11R6/lib64   /usr/X11R6/include
  /usr/X11R7/lib   /usr/X11R7/lib64   /usr/X11R7/include
)

#
# Library prefixes, suffixes, extra libs.
#
set(CMAKE_LINK_LIBRARY_SUFFIX "")
set(CMAKE_STATIC_LIBRARY_PREFIX "lib")     # lib
set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")      # .a

set(CMAKE_SHARED_LIBRARY_PREFIX "lib")     # lib
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")     # .so
set(CMAKE_EXECUTABLE_SUFFIX "")            # .exe
set(CMAKE_DL_LIBS "dl")

#
# This macro needs to be called for dynamic library support.  Unfortunately on BGP,
# We can't support both static and dynamic links in the same platform file.  The
# dynamic link platform file needs to call this explicitly to set up dynamic linking.
#
macro(__BlueGeneP_set_dynamic_flags compiler_id lang)
  if (${compiler_id} STREQUAL XL)
    # Flags for XL compilers if we explicitly detected XL
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIC            "-qpic")
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIE            "-qpie")
    set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS           "-qpic")
    set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS    "-qmkshrobj -qnostaticlink")
    set(BGP_${lang}_DYNAMIC_EXE_FLAGS                "-qnostaticlink -qnostaticlink=libgcc")
  else()
    # Assume flags for GNU compilers (if the ID is GNU *or* anything else).
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIC            "-fPIC")
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIE            "-fPIE")
    set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS           "-fPIC")
    set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS    "-shared")
    set(BGP_${lang}_DYNAMIC_EXE_FLAGS                "-dynamic")
  endif()

  # Both toolchains use the GNU linker on BG/P, so these options are shared.
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG      "-Wl,-rpath,")
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_${lang}_FLAG   "-Wl,-rpath-link,")
  set(CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG       "-Wl,-soname,")
  set(CMAKE_EXE_EXPORTS_${lang}_FLAG                 "-Wl,--export-dynamic")
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS        "")  # +s, flag for exe link to use shared lib
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP  ":") # : or empty

  set(BGP_${lang}_DEFAULT_EXE_FLAGS
    "<FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
  set(CMAKE_${lang}_LINK_EXECUTABLE
    "<CMAKE_${lang}_COMPILER> -Wl,-relax ${BGP_${lang}_DYNAMIC_EXE_FLAGS} ${BGP_${lang}_DEFAULT_EXE_FLAGS}")
endmacro()

#
# This macro needs to be called for static builds.  Right now it just adds -Wl,-relax
# to the link line.
#
macro(__BlueGeneP_set_static_flags compiler_id lang)
  set(BGP_${lang}_DEFAULT_EXE_FLAGS
    "<FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
  set(CMAKE_${lang}_LINK_EXECUTABLE
    "<CMAKE_${lang}_COMPILER> -Wl,-relax ${BGP_${lang}_DEFAULT_EXE_FLAGS}")
endmacro()
