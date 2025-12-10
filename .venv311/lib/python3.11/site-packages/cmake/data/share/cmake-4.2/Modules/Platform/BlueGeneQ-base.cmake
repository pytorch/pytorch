# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


#
# Blue Gene/Q base platform file.
#
# NOTE: Do not set your platform to "BlueGeneQ-base".  This file is
# included by the real platform files.  Use one of these two platforms
# instead:
#
#     BlueGeneQ-dynamic  For dynamically linked executables
#     BlueGeneQ-static   For statically linked executables
#
# The platform you choose doesn't affect whether or not you can build
# shared or static libraries -- it ONLY changs whether exeuatbles are linked
# statically or dynamically.
#
# This platform file tries its best to adhere to the behavior of the MPI
# compiler wrappers included with the latest BG/P drivers.
#

#
# This adds directories that find commands should specifically ignore
# for cross compiles.  Most of these directories are the includeand
# lib directories for the frontend on BG/P systems.  Not ignoring
# these can cause things like FindX11 to find a frontend PPC version
# mistakenly.  We use this on BG instead of re-rooting because backend
# libraries are typically strewn about the filesystem, and we can't
# re-root ALL backend libraries to a single place.
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
# BG/Q supports dynamic libraries regardless of whether we're building
# static or dynamic *executables*.
#
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
set(CMAKE_FIND_LIBRARY_PREFIXES "lib")

#
# For BGQ builds, we're cross compiling, but we don't want to re-root things
# (e.g. with CMAKE_FIND_ROOT_PATH) because users may have libraries anywhere on
# the shared filesystems, and this may lie outside the root.  Instead, we set the
# system directories so that the various system BG CNK library locations are
# searched first.  This is not the clearest thing in the world, given IBM's driver
# layout, but this should cover all the standard ones.
#
macro(__BlueGeneQ_common_setup compiler_id lang)
  # Need to use the version of the comm lib compiled with the right compiler.
  set(__BlueGeneQ_commlib_dir gcc)
  if (${compiler_id} STREQUAL XL)
    set(__BlueGeneQ_commlib_dir xl)
  endif()

  set(CMAKE_SYSTEM_LIBRARY_PATH
    /bgsys/drivers/ppcfloor/comm/default/lib                    # default comm layer (used by mpi compiler wrappers)
    /bgsys/drivers/ppcfloor/comm/${__BlueGeneQ_commlib_dir}/lib # PAMI, other lower-level comm libraries
    /bgsys/drivers/ppcfloor/gnu-linux/lib                       # CNK python installation directory
    /bgsys/drivers/ppcfloor/gnu-linux/powerpc64-bgq-linux/lib   # CNK Linux image -- standard runtime libs, pthread, etc.
    )

  # Add all the system include paths.
  set(CMAKE_SYSTEM_INCLUDE_PATH
    /bgsys/drivers/ppcfloor/comm/sys/include
    /bgsys/drivers/ppcfloor/
    /bgsys/drivers/ppcfloor/spi/include
    /bgsys/drivers/ppcfloor/spi/include/kernel/cnk
    /bgsys/drivers/ppcfloor/comm/${__BlueGeneQ_commlib_dir}/include
    )

  # Ensure that the system directories are included with the regular compilers, as users will expect this
  # to do the same thing as the MPI compilers, which add these flags.
  set(BGQ_SYSTEM_INCLUDES "")
  foreach(dir ${CMAKE_SYSTEM_INCLUDE_PATH})
    string(APPEND BGQ_SYSTEM_INCLUDES " -I${dir}")
  endforeach()
  set(CMAKE_C_COMPILE_OBJECT   "<CMAKE_C_COMPILER> <DEFINES> ${BGQ_SYSTEM_INCLUDES} <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
  set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES> ${BGQ_SYSTEM_INCLUDES} <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")

  #
  # Code below does setup for shared libraries.  That this is done
  # regardless of whether the platform is static or dynamic -- you can make
  # shared libraries even if you intend to make static executables, you just
  # can't make a dynamic executable if you use the static platform file.
  #
  if (${compiler_id} STREQUAL XL)
    # Flags for XL compilers if we explicitly detected XL
    set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS           "-qpic")
    set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS    "-qmkshrobj -qnostaticlink")
  else()
    # Assume flags for GNU compilers (if the ID is GNU *or* anything else).
    set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS           "-fPIC")
    set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS    "-shared")
  endif()

  # Both toolchains use the GNU linker on BG/P, so these options are shared.
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG      "-Wl,-rpath,")
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_${lang}_FLAG   "-Wl,-rpath-link,")
  set(CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG       "-Wl,-soname,")
  set(CMAKE_EXE_EXPORTS_${lang}_FLAG                 "-Wl,--export-dynamic")
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS        "")  # +s, flag for exe link to use shared lib
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP  ":") # : or empty

endmacro()

#
# This macro needs to be called for dynamic library support.  Unfortunately on BG,
# We can't support both static and dynamic links in the same platform file.  The
# dynamic link platform file needs to call this explicitly to set up dynamic linking.
#
macro(__BlueGeneQ_setup_dynamic compiler_id lang)
  __BlueGeneQ_common_setup(${compiler_id} ${lang})

  if (${compiler_id} STREQUAL XL)
    set(BGQ_${lang}_DYNAMIC_EXE_FLAGS "-qnostaticlink -qnostaticlink=libgcc")
  else()
    set(BGQ_${lang}_DYNAMIC_EXE_FLAGS "-dynamic")
  endif()

  # For dynamic executables, need to provide special BG/Q arguments.
  set(BGQ_${lang}_DEFAULT_EXE_FLAGS
    "<FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
  set(CMAKE_${lang}_LINK_EXECUTABLE
    "<CMAKE_${lang}_COMPILER> -Wl,-relax ${BGQ_${lang}_DYNAMIC_EXE_FLAGS} ${BGQ_${lang}_DEFAULT_EXE_FLAGS}")
endmacro()

#
# This macro needs to be called for static builds.  Right now it just adds -Wl,-relax
# to the link line.
#
macro(__BlueGeneQ_setup_static compiler_id lang)
  __BlueGeneQ_common_setup(${compiler_id} ${lang})

  # For static executables, use default link settings.
  set(BGQ_${lang}_DEFAULT_EXE_FLAGS
    "<FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
  set(CMAKE_${lang}_LINK_EXECUTABLE
    "<CMAKE_${lang}_COMPILER> -Wl,-relax ${BGQ_${lang}_DEFAULT_EXE_FLAGS}")
endmacro()
