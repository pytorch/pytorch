# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__WINDOWS_GNU)
  return()
endif()
set(__WINDOWS_GNU 1)

set(MINGW 1)

# On Windows hosts, in MSYSTEM environments, search standard prefixes.
if(CMAKE_HOST_WIN32)
  # Bootstrap CMake does not have cmake_host_system_information.
  if(COMMAND cmake_host_system_information)
    cmake_host_system_information(RESULT _MSYSTEM_PREFIX QUERY MSYSTEM_PREFIX)
  elseif(IS_DIRECTORY "$ENV{MSYSTEM_PREFIX}")
    set(_MSYSTEM_PREFIX "$ENV{MSYSTEM_PREFIX}")
  endif()

  # Search this MSYSTEM environment's equivalent to /usr/local and /usr.
  if(_MSYSTEM_PREFIX)
    list(PREPEND CMAKE_SYSTEM_PREFIX_PATH "${_MSYSTEM_PREFIX}")
    if(IS_DIRECTORY "${_MSYSTEM_PREFIX}/local")
      list(PREPEND CMAKE_SYSTEM_PREFIX_PATH "${_MSYSTEM_PREFIX}/local")
    endif()
  endif()
  unset(_MSYSTEM_PREFIX)
endif()

set(CMAKE_IMPORT_LIBRARY_PREFIX "lib")
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
set(CMAKE_SHARED_MODULE_PREFIX  "lib")
set(CMAKE_STATIC_LIBRARY_PREFIX "lib")

set(CMAKE_EXECUTABLE_SUFFIX     ".exe")
set(CMAKE_IMPORT_LIBRARY_SUFFIX ".dll.a")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
set(CMAKE_SHARED_MODULE_SUFFIX  ".dll")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")

set(CMAKE_EXTRA_LINK_EXTENSIONS ".lib") # MinGW can also link to a MS .lib

set(CMAKE_FIND_LIBRARY_PREFIXES "lib" "")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll.a" ".a" ".lib")
set(CMAKE_C_STANDARD_LIBRARIES_INIT "-lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32")
set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_C_STANDARD_LIBRARIES_INIT}")

set(CMAKE_DL_LIBS "")
set(CMAKE_LIBRARY_PATH_FLAG "-L")
set(CMAKE_LINK_LIBRARY_FLAG "-l")
set(CMAKE_LINK_DEF_FILE_FLAG "") # Empty string: passing the file is enough
set(CMAKE_LINK_LIBRARY_SUFFIX "")

set(CMAKE_GNULD_IMAGE_VERSION
  "-Wl,--major-image-version,<TARGET_VERSION_MAJOR>,--minor-image-version,<TARGET_VERSION_MINOR>")

# Check if GNU ld is too old to support @FILE syntax.
set(__WINDOWS_GNU_LD_RESPONSE 1)
execute_process(COMMAND ld -v OUTPUT_VARIABLE _help ERROR_VARIABLE _help)
if("${_help}" MATCHES "GNU ld .* 2\\.1[1-6]")
  set(__WINDOWS_GNU_LD_RESPONSE 0)
endif()


# Features for LINK_GROUP generator expression
## RESCAN: request the linker to rescan static libraries until there is
## no pending undefined symbols
set(CMAKE_LINK_GROUP_USING_RESCAN "LINKER:--start-group" "LINKER:--end-group")
set(CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED TRUE)


macro(__windows_compiler_gnu lang)

  # Create archiving rules to support large object file lists for static libraries.
  set(CMAKE_${lang}_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")

  # Initialize C link type selection flags.  These flags are used when
  # building a shared library, shared module, or executable that links
  # to other libraries to select whether to use the static or shared
  # versions of the libraries.
  foreach(type SHARED_LIBRARY SHARED_MODULE EXE)
    set(CMAKE_${type}_LINK_STATIC_${lang}_FLAGS "-Wl,-Bstatic")
    set(CMAKE_${type}_LINK_DYNAMIC_${lang}_FLAGS "-Wl,-Bdynamic")
  endforeach()

  set(CMAKE_${lang}_VERBOSE_LINK_FLAG "-Wl,-v")

  # linker selection
  set(CMAKE_${lang}_USING_LINKER_SYSTEM "")
  set(CMAKE_${lang}_USING_LINKER_BFD "-fuse-ld=bfd")
  set(CMAKE_${lang}_USING_LINKER_LLD "-fuse-ld=lld")

  # No -fPIC on Windows
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "")
  set(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER NO)
  set(CMAKE_${lang}_LINK_OPTIONS_PIE "")
  set(CMAKE_${lang}_LINK_OPTIONS_NO_PIE "")
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "")

  set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_OBJECTS ${__WINDOWS_GNU_LD_RESPONSE})
  set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_LIBRARIES ${__WINDOWS_GNU_LD_RESPONSE})
  set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_INCLUDES 1)

  # We prefer "@" for response files but it is not supported by gcc 3.
  execute_process(COMMAND ${CMAKE_${lang}_COMPILER} --version OUTPUT_VARIABLE _ver ERROR_VARIABLE _ver)
  if("${_ver}" MATCHES "\\(GCC\\) 3\\.")
    if("${lang}" STREQUAL "Fortran")
      # The GNU Fortran compiler reports an error:
      #   no input files; unwilling to write output files
      # when the response file is passed with "-Wl,@".
      set(CMAKE_Fortran_USE_RESPONSE_FILE_FOR_OBJECTS 0)
    else()
      # Use "-Wl,@" to pass the response file to the linker.
      set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "-Wl,@")
    endif()
    # The GNU 3.x compilers do not support response files (only linkers).
    set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_INCLUDES 0)
    # Link libraries are generated only for the front-end.
    set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
  else()
    # Use "@" to pass the response file to the front-end.
    set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "@")
  endif()

  set(CMAKE_${lang}_LINK_DEF_FILE_FLAG "") # Empty string: passing the file is enough

  # Binary link rules.
  set(CMAKE_${lang}_CREATE_SHARED_MODULE
    "<CMAKE_${lang}_COMPILER> <CMAKE_SHARED_MODULE_${lang}_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> -o <TARGET> ${CMAKE_GNULD_IMAGE_VERSION} <OBJECTS> <LINK_LIBRARIES>")
  set(CMAKE_${lang}_CREATE_SHARED_LIBRARY
    "<CMAKE_${lang}_COMPILER> <CMAKE_SHARED_LIBRARY_${lang}_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> -o <TARGET> -Wl,--out-implib,<TARGET_IMPLIB> ${CMAKE_GNULD_IMAGE_VERSION} <OBJECTS> <LINK_LIBRARIES>")
  set(CMAKE_${lang}_LINK_EXECUTABLE
    "<CMAKE_${lang}_COMPILER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> -Wl,--out-implib,<TARGET_IMPLIB> ${CMAKE_GNULD_IMAGE_VERSION} <LINK_LIBRARIES>")
  set(CMAKE_${lang}_CREATE_WIN32_EXE "-mwindows")

  list(APPEND CMAKE_${lang}_ABI_FILES "Platform/Windows-GNU-${lang}-ABI")

  # Support very long lists of object files.
  # TODO: check for which gcc versions this is still needed, not needed for gcc >= 4.4.
  # Ninja generator doesn't support this work around.
  if("${CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG}" STREQUAL "@" AND NOT CMAKE_GENERATOR MATCHES "Ninja")
    foreach(rule CREATE_SHARED_MODULE CREATE_SHARED_LIBRARY LINK_EXECUTABLE)
      # The gcc/collect2/ld toolchain does not use response files
      # internally so we cannot pass long object lists.  Instead pass
      # the object file list in a response file to the archiver to put
      # them in a temporary archive.  Hand the archive to the linker.
      string(REPLACE "<OBJECTS>" "-Wl,--whole-archive <OBJECT_DIR>/objects.a -Wl,--no-whole-archive"
        CMAKE_${lang}_${rule} "${CMAKE_${lang}_${rule}}")
      set(CMAKE_${lang}_${rule}
        "<CMAKE_COMMAND> -E rm -f <OBJECT_DIR>/objects.a"
        "<CMAKE_AR> qc <OBJECT_DIR>/objects.a <OBJECTS>"
        "${CMAKE_${lang}_${rule}}"
        )
    endforeach()
  endif()

  if(NOT CMAKE_RC_COMPILER_INIT AND NOT CMAKE_GENERATOR_RC)
    set(_CMAKE_RC_COMPILER_LIST ${_CMAKE_TOOLCHAIN_PREFIX}windres windres)
    set(_CMAKE_RC_COMPILER_FALLBACK windres)
  endif()

  enable_language(RC)
endmacro()

macro(__windows_compiler_gnu_abi lang)
  if(CMAKE_NO_GNUtoMS)
    set(CMAKE_GNUtoMS 0)
  else()
    option(CMAKE_GNUtoMS "Convert GNU import libraries to MS format (requires Visual Studio)" OFF)
  endif()

  if(CMAKE_GNUtoMS AND NOT CMAKE_GNUtoMS_LIB)
    # Find MS development environment setup script for this architecture.
    # We need to use the MS Librarian tool (lib.exe).
    # Find the most recent version available.

    # Query the VS Installer tool for locations of VS 2017 and above.
    set(_vs_installer_paths "")
    foreach(vs RANGE 18 15 -1) # change the first number to the largest supported version
      cmake_host_system_information(RESULT _vs_dir QUERY VS_${vs}_DIR)
      if(_vs_dir)
        list(APPEND _vs_installer_paths "${_vs_dir}/VC/Auxiliary/Build")
      endif()
    endforeach()

    if("${CMAKE_SIZEOF_VOID_P}" EQUAL 4)
      find_program(CMAKE_GNUtoMS_VCVARS NAMES vcvars32.bat
        DOC "Visual Studio vcvars32.bat"
        PATHS
        ${_vs_installer_paths}
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\14.0\\Setup\\VC;ProductDir]/bin"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\12.0\\Setup\\VC;ProductDir]/bin"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\11.0\\Setup\\VC;ProductDir]/bin"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\10.0\\Setup\\VC;ProductDir]/bin"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\9.0\\Setup\\VC;ProductDir]/bin"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\8.0\\Setup\\VC;ProductDir]/bin"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\7.1\\Setup\\VC;ProductDir]/bin"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\6.0\\Setup\\Microsoft Visual C++;ProductDir]/bin"
        )
      set(CMAKE_GNUtoMS_ARCH x86)
    elseif("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
      find_program(CMAKE_GNUtoMS_VCVARS NAMES vcvars64.bat vcvarsamd64.bat
        DOC "Visual Studio vcvarsamd64.bat"
        PATHS
        ${_vs_installer_paths}
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\14.0\\Setup\\VC;ProductDir]/bin/amd64"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\12.0\\Setup\\VC;ProductDir]/bin/amd64"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\11.0\\Setup\\VC;ProductDir]/bin/amd64"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\10.0\\Setup\\VC;ProductDir]/bin/amd64"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\9.0\\Setup\\VC;ProductDir]/bin/amd64"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\8.0\\Setup\\VC;ProductDir]/bin/amd64"
        )
      set(CMAKE_GNUtoMS_ARCH amd64)
    endif()
    unset(_vs_installer_paths)
    set_property(CACHE CMAKE_GNUtoMS_VCVARS PROPERTY ADVANCED 1)
    if(CMAKE_GNUtoMS_VCVARS)
      # Create helper script to run lib.exe from MS environment.
      string(REPLACE "/" "\\" CMAKE_GNUtoMS_BAT "${CMAKE_GNUtoMS_VCVARS}")
      set(CMAKE_GNUtoMS_LIB ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeGNUtoMS_lib.bat)
      configure_file(${CMAKE_ROOT}/Modules/Platform/GNUtoMS_lib.bat.in ${CMAKE_GNUtoMS_LIB})
    else()
      message(WARNING "Disabling CMAKE_GNUtoMS option because CMAKE_GNUtoMS_VCVARS is not set.")
      set(CMAKE_GNUtoMS 0)
    endif()
  endif()

  if(CMAKE_GNUtoMS)
    # Teach CMake how to create a MS import library at link time.
    set(CMAKE_${lang}_GNUtoMS_RULE " -Wl,--output-def,<TARGET_NAME>.def"
      "<CMAKE_COMMAND> -Dlib=\"${CMAKE_GNUtoMS_LIB}\" -Ddef=<TARGET_NAME>.def -Ddll=<TARGET> -Dimp=<TARGET_IMPLIB> -P \"${CMAKE_ROOT}/Modules/Platform/GNUtoMS_lib.cmake\""
      )
  endif()
endmacro()
