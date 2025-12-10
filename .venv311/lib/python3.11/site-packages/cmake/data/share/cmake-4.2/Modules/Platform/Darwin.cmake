if(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "tvOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS" OR CMAKE_SYSTEM_NAME STREQUAL "watchOS")
  if(NOT DEFINED CMAKE_MACOSX_BUNDLE)
    set(CMAKE_MACOSX_BUNDLE ON)
  endif()

  list(APPEND CMAKE_FIND_ROOT_PATH "${_CMAKE_OSX_SYSROOT_PATH}")
  if(NOT DEFINED CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
      set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  endif()
  if(NOT DEFINED CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
      set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
  endif()
  if(NOT DEFINED CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
      set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
  endif()
endif()

# Darwin versions:
#   6.x == Mac OSX 10.2 (Jaguar)
#   7.x == Mac OSX 10.3 (Panther)
#   8.x == Mac OSX 10.4 (Tiger)
#   9.x == Mac OSX 10.5 (Leopard)
#  10.x == Mac OSX 10.6 (Snow Leopard)
#  11.x == Mac OSX 10.7 (Lion)
#  12.x == Mac OSX 10.8 (Mountain Lion)
string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*$" "\\1" DARWIN_MAJOR_VERSION "${CMAKE_SYSTEM_VERSION}")
string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*$" "\\2" DARWIN_MINOR_VERSION "${CMAKE_SYSTEM_VERSION}")

# Do not use the "-Wl,-search_paths_first" flag with the OSX 10.2 compiler.
# Done this way because it is too early to do a TRY_COMPILE.
if(NOT DEFINED HAVE_FLAG_SEARCH_PATHS_FIRST)
  set(HAVE_FLAG_SEARCH_PATHS_FIRST 0)
  if("${DARWIN_MAJOR_VERSION}" GREATER 6)
    set(HAVE_FLAG_SEARCH_PATHS_FIRST 1)
  endif()
endif()
# More desirable, but does not work:
  #include(CheckCXXCompilerFlag)
  #check_cxx_compiler_flag("-Wl,-search_paths_first" HAVE_FLAG_SEARCH_PATHS_FIRST)

set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
set(CMAKE_EXTRA_SHARED_LIBRARY_SUFFIXES ".tbd" ".so")
set(CMAKE_SHARED_MODULE_PREFIX "lib")
set(CMAKE_SHARED_MODULE_SUFFIX ".so")
set(CMAKE_APPLE_IMPORT_FILE_PREFIX "lib")
set(CMAKE_APPLE_IMPORT_FILE_SUFFIX ".tbd")
set(CMAKE_MODULE_EXISTS 1)
set(CMAKE_DL_LIBS "")
if(NOT (DEFINED _CMAKE_HOST_OSX_VERSION AND _CMAKE_HOST_OSX_VERSION VERSION_LESS "10.5"))
  set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-rpath,")
endif()

foreach(lang C CXX OBJC OBJCXX)
  set(CMAKE_${lang}_OSX_COMPATIBILITY_VERSION_FLAG "-compatibility_version ")
  set(CMAKE_${lang}_OSX_CURRENT_VERSION_FLAG "-current_version ")
  set(CMAKE_${lang}_LINK_FLAGS "-Wl,-headerpad_max_install_names")

  if(HAVE_FLAG_SEARCH_PATHS_FIRST)
    set(CMAKE_${lang}_LINK_FLAGS "-Wl,-search_paths_first ${CMAKE_${lang}_LINK_FLAGS}")
  endif()

  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-dynamiclib -Wl,-headerpad_max_install_names")
  set(CMAKE_SHARED_MODULE_CREATE_${lang}_FLAGS "-bundle -Wl,-headerpad_max_install_names")
  set(CMAKE_SHARED_MODULE_LOADER_${lang}_FLAG "-Wl,-bundle_loader,")
endforeach()

set(CMAKE_PLATFORM_HAS_INSTALLNAME 1)
set(CMAKE_FIND_LIBRARY_SUFFIXES ".tbd" ".dylib" ".so" ".a")

# hack: if a new cmake (which uses CMAKE_INSTALL_NAME_TOOL) runs on an old build tree
# (where install_name_tool was hardcoded) and where CMAKE_INSTALL_NAME_TOOL isn't in the cache
# and still cmake didn't fail in CMakeFindBinUtils.cmake (because it isn't rerun)
# hardcode CMAKE_INSTALL_NAME_TOOL here to install_name_tool, so it behaves as it did before, Alex
if(NOT DEFINED CMAKE_INSTALL_NAME_TOOL)
  find_program(CMAKE_INSTALL_NAME_TOOL install_name_tool)
  mark_as_advanced(CMAKE_INSTALL_NAME_TOOL)
endif()

# Enable shared library versioning.
set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-install_name")

if(DEFINED _CMAKE_HOST_OSX_VERSION AND _CMAKE_HOST_OSX_VERSION VERSION_LESS "10.5")
  # Need to list dependent shared libraries on link line.  When building
  # with -isysroot (for universal binaries), the linker always looks for
  # dependent libraries under the sysroot.  Listing them on the link
  # line works around the problem.
  set(CMAKE_LINK_DEPENDENT_LIBRARY_FILES 1)
endif()

foreach(lang C CXX Fortran OBJC OBJCXX)
  # Xcode does not support -isystem yet.
  if(XCODE)
    set(CMAKE_INCLUDE_SYSTEM_FLAG_${lang})
  endif()

  set(CMAKE_${lang}_CREATE_SHARED_LIBRARY
    "<CMAKE_${lang}_COMPILER> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> -o <TARGET> <SONAME_FLAG> <TARGET_INSTALLNAME_DIR><TARGET_SONAME> <OBJECTS> <LINK_LIBRARIES>")

  set(CMAKE_${lang}_CREATE_SHARED_MODULE
      "<CMAKE_${lang}_COMPILER> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

  set(CMAKE_${lang}_CREATE_MACOSX_FRAMEWORK
      "<CMAKE_${lang}_COMPILER> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> -o <TARGET> <SONAME_FLAG> <TARGET_INSTALLNAME_DIR><TARGET_SONAME> <OBJECTS> <LINK_LIBRARIES>")

  # Set default framework search path flag for languages known to use a
  # preprocessor that may find headers in frameworks.
  set(CMAKE_${lang}_FRAMEWORK_SEARCH_FLAG -F)
endforeach()

# To generate text-based stubs
set(CMAKE_CREATE_TEXT_STUBS "<CMAKE_TAPI> stubify -isysroot <CMAKE_OSX_SYSROOT> -o <TARGET_IMPLIB> <TARGET>")

# Defines LINK_LIBRARY features for frameworks
set(CMAKE_LINK_LIBRARY_USING_FRAMEWORK "LINKER:-framework,<LIBRARY>")
set(CMAKE_LINK_LIBRARY_USING_FRAMEWORK_SUPPORTED TRUE)
set(CMAKE_LINK_LIBRARY_FRAMEWORK_ATTRIBUTES LIBRARY_TYPE=STATIC,SHARED DEDUPLICATION=DEFAULT OVERRIDE=DEFAULT)

set(CMAKE_LINK_LIBRARY_USING_NEEDED_FRAMEWORK "LINKER:-needed_framework,<LIBRARY>")
set(CMAKE_LINK_LIBRARY_USING_NEEDED_FRAMEWORK_SUPPORTED TRUE)
set(CMAKE_LINK_LIBRARY_NEEDED_FRAMEWORK_ATTRIBUTES LIBRARY_TYPE=STATIC,SHARED DEDUPLICATION=DEFAULT OVERRIDE=DEFAULT)

set(CMAKE_LINK_LIBRARY_USING_REEXPORT_FRAMEWORK "LINKER:-reexport_framework,<LIBRARY>")
set(CMAKE_LINK_LIBRARY_USING_REEXPORT_FRAMEWORK_SUPPORTED TRUE)
set(CMAKE_LINK_LIBRARY_REEXPORT_FRAMEWORK_ATTRIBUTES LIBRARY_TYPE=STATIC,SHARED DEDUPLICATION=DEFAULT OVERRIDE=DEFAULT)

set(CMAKE_LINK_LIBRARY_USING_WEAK_FRAMEWORK "LINKER:-weak_framework,<LIBRARY>")
set(CMAKE_LINK_LIBRARY_USING_WEAK_FRAMEWORK_SUPPORTED TRUE)
set(CMAKE_LINK_LIBRARY_WEAK_FRAMEWORK_ATTRIBUTES LIBRARY_TYPE=STATIC,SHARED DEDUPLICATION=DEFAULT OVERRIDE=DEFAULT)

# Defines LINK_LIBRARY features for libraries
set(CMAKE_LINK_LIBRARY_USING_NEEDED_LIBRARY "PATH{LINKER:-needed_library,<LIBRARY>}NAME{LINKER:-needed-l<LIBRARY>}")
set(CMAKE_LINK_LIBRARY_USING_NEEDED_LIBRARY_SUPPORTED TRUE)
set(CMAKE_LINK_LIBRARY_NEEDED_LIBRARY_ATTRIBUTES LIBRARY_TYPE=SHARED DEDUPLICATION=DEFAULT OVERRIDE=DEFAULT)

set(CMAKE_LINK_LIBRARY_USING_REEXPORT_LIBRARY "PATH{LINKER:-reexport_library,<LIBRARY>}NAME{LINKER:-reexport-l<LIBRARY>}")
set(CMAKE_LINK_LIBRARY_USING_REEXPORT_LIBRARY_SUPPORTED TRUE)
set(CMAKE_LINK_LIBRARY_REEXPORT_LIBRARY_ATTRIBUTES LIBRARY_TYPE=STATIC,SHARED DEDUPLICATION=DEFAULT OVERRIDE=DEFAULT)

set(CMAKE_LINK_LIBRARY_USING_WEAK_LIBRARY "PATH{LINKER:-weak_library,<LIBRARY>}NAME{LINKER:-weak-l<LIBRARY>}")
set(CMAKE_LINK_LIBRARY_USING_WEAK_LIBRARY_SUPPORTED TRUE)
set(CMAKE_LINK_LIBRARY_WEAK_LIBRARY_ATTRIBUTES LIBRARY_TYPE=STATIC,SHARED DEDUPLICATION=DEFAULT OVERRIDE=DEFAULT)

# default to searching for frameworks first
if(NOT DEFINED CMAKE_FIND_FRAMEWORK)
  set(CMAKE_FIND_FRAMEWORK FIRST)
endif()

# Older OS X linkers do not report their framework search path
# with -v but "man ld" documents the following locations.
set(CMAKE_PLATFORM_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES
  ${_CMAKE_OSX_SYSROOT_PATH}/Library/Frameworks
  ${_CMAKE_OSX_SYSROOT_PATH}/System/Library/Frameworks
  )
if(_CMAKE_OSX_SYSROOT_PATH)
  # Treat some paths as implicit so we do not override the SDK versions.
  list(APPEND CMAKE_PLATFORM_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES
    /System/Library/Frameworks)
endif()

if(DEFINED _CMAKE_HOST_OSX_VERSION AND _CMAKE_HOST_OSX_VERSION VERSION_LESS "10.5")
  # Older OS X tools had more implicit paths.
  list(APPEND CMAKE_PLATFORM_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES
    ${_CMAKE_OSX_SYSROOT_PATH}/Network/Library/Frameworks)
endif()

# set up the default search directories for frameworks
set(CMAKE_SYSTEM_FRAMEWORK_PATH
  ~/Library/Frameworks
  )
if(_CMAKE_OSX_SYSROOT_PATH)
  list(APPEND CMAKE_SYSTEM_FRAMEWORK_PATH
    ${_CMAKE_OSX_SYSROOT_PATH}/Library/Frameworks
    ${_CMAKE_OSX_SYSROOT_PATH}/Network/Library/Frameworks
    ${_CMAKE_OSX_SYSROOT_PATH}/System/Library/Frameworks
    )
  # add platform developer framework path if exists
  foreach(_path
    # Xcode 6
    ${_CMAKE_OSX_SYSROOT_PATH}/../../Library/Frameworks
    # Xcode 5 iOS
    ${_CMAKE_OSX_SYSROOT_PATH}/Developer/Library/Frameworks
    # Xcode 5 OSX
    ${_CMAKE_OSX_SYSROOT_PATH}/../../../../../Library/Frameworks
    )
    get_filename_component(_absolute_path "${_path}" ABSOLUTE)
    if(EXISTS "${_absolute_path}")
      list(APPEND CMAKE_SYSTEM_FRAMEWORK_PATH "${_absolute_path}")
      break()
    endif()
  endforeach()

  if(EXISTS ${_CMAKE_OSX_SYSROOT_PATH}/usr/lib)
    list(INSERT CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES 0 ${_CMAKE_OSX_SYSROOT_PATH}/usr/lib)
  endif()
  if(EXISTS ${_CMAKE_OSX_SYSROOT_PATH}/usr/local/lib)
    list(INSERT CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES 0 ${_CMAKE_OSX_SYSROOT_PATH}/usr/local/lib)
  endif()
endif()
if (OSX_DEVELOPER_ROOT AND EXISTS "${OSX_DEVELOPER_ROOT}/Library/Frameworks")
  list(APPEND CMAKE_SYSTEM_FRAMEWORK_PATH
    ${OSX_DEVELOPER_ROOT}/Library/Frameworks)
endif()
list(APPEND CMAKE_SYSTEM_FRAMEWORK_PATH
  /Library/Frameworks
  /Network/Library/Frameworks
  /System/Library/Frameworks)

# Warn about known system misconfiguration case.
if(CMAKE_OSX_SYSROOT)
  get_property(_IN_TC GLOBAL PROPERTY IN_TRY_COMPILE)
  if(NOT _IN_TC AND
     NOT IS_SYMLINK "${CMAKE_OSX_SYSROOT}/Library/Frameworks"
     AND IS_SYMLINK "${CMAKE_OSX_SYSROOT}/Library/Frameworks/Frameworks")
    message(WARNING "The SDK Library/Frameworks path\n"
      " ${CMAKE_OSX_SYSROOT}/Library/Frameworks\n"
      "is not set up correctly on this system.  "
      "This is known to occur when installing Xcode 3.2.6:\n"
      " http://bugs.python.org/issue14018\n"
      "The problem may cause build errors that report missing system frameworks.  "
      "Fix your SDK symlinks to resolve this issue and avoid this warning."
      )
  endif()
endif()

# default to searching for application bundles first
if(NOT DEFINED CMAKE_FIND_APPBUNDLE)
  set(CMAKE_FIND_APPBUNDLE FIRST)
endif()
# set up the default search directories for application bundles
set(_apps_paths)
foreach(_path
  "~/Applications"
  "/Applications"
  "${OSX_DEVELOPER_ROOT}/../Applications" # Xcode 4.3+
  "${OSX_DEVELOPER_ROOT}/Applications"    # pre-4.3
  )
  get_filename_component(_apps "${_path}" ABSOLUTE)
  if(EXISTS "${_apps}")
    list(APPEND _apps_paths "${_apps}")
  endif()
endforeach()
if(_apps_paths)
  list(REMOVE_DUPLICATES _apps_paths)
endif()
set(CMAKE_SYSTEM_APPBUNDLE_PATH
  ${_apps_paths})
unset(_apps_paths)

include(Platform/UnixPaths)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  execute_process(
    COMMAND brew --prefix
    OUTPUT_VARIABLE _cmake_homebrew_prefix
    RESULT_VARIABLE _brew_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (_brew_result EQUAL 0 AND IS_DIRECTORY "${_cmake_homebrew_prefix}")
    list(PREPEND CMAKE_SYSTEM_PREFIX_PATH "${_cmake_homebrew_prefix}")
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    list(PREPEND CMAKE_SYSTEM_PREFIX_PATH
      /opt/homebrew # Brew on Apple Silicon
      )
  else()
    list(PREPEND CMAKE_SYSTEM_PREFIX_PATH
      /usr/local # Brew on Intel
      )
  endif()
  unset(_cmake_homebrew_prefix)
  unset(_brew_result)
endif()

if(_CMAKE_OSX_SYSROOT_PATH)
  if(EXISTS ${_CMAKE_OSX_SYSROOT_PATH}/usr/include)
    list(INSERT CMAKE_SYSTEM_PREFIX_PATH 0 ${_CMAKE_OSX_SYSROOT_PATH}/usr)
    foreach(lang C CXX OBJC OBJCXX Swift)
      list(APPEND _CMAKE_${lang}_IMPLICIT_INCLUDE_DIRECTORIES_INIT ${_CMAKE_OSX_SYSROOT_PATH}/usr/include)
    endforeach()
  endif()
  if(EXISTS ${_CMAKE_OSX_SYSROOT_PATH}/usr/local/include)
    list(INSERT CMAKE_SYSTEM_PREFIX_PATH 0 ${_CMAKE_OSX_SYSROOT_PATH}/usr/local)
    foreach(lang C CXX OBJC OBJCXX Swift)
      list(APPEND _CMAKE_${lang}_IMPLICIT_INCLUDE_DIRECTORIES_INIT ${_CMAKE_OSX_SYSROOT_PATH}/usr/local/include)
    endforeach()
  endif()
endif()
list(APPEND CMAKE_SYSTEM_PREFIX_PATH
  /sw        # Fink
  /opt/local # MacPorts
  )
