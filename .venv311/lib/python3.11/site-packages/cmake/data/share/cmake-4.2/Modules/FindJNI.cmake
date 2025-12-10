# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindJNI
-------

Finds the Java Native Interface (JNI) include directories and libraries:

.. code-block:: cmake

  find_package(JNI [<version>] [COMPONENTS <components>...] [...])

JNI enables Java code running in a Java Virtual Machine (JVM) or Dalvik Virtual
Machine (DVM) on Android to call and be called by native applications and
libraries written in other languages such as C and C++.

This module finds if Java is installed and determines where the
include files and libraries are.  It also determines what the name of
the library is.

.. versionadded:: 3.24

  Imported targets, components, and Android NDK support.

  When using Android NDK, the corresponding package version is reported and a
  specific release can be requested. At Android API level 31 and above, the
  additional ``NativeHelper`` component can be requested. ``NativeHelper`` is
  also exposed as an implicit dependency of the ``JVM`` component (only if this
  does not cause a conflict) which provides a uniform access to JVM functions.

Components
^^^^^^^^^^

This module supports optional components, which can be specified with the
:command:`find_package` command:

.. code-block:: cmake

  find_package(JNI [COMPONENTS <components>...])

Supported components include:

``AWT``
  .. versionadded:: 3.24

  Finds the Java Abstract Window Toolkit (AWT).

``JVM``
  .. versionadded:: 3.24

  Finds the Java Virtual Machine (JVM).

``NativeHelper``
  .. versionadded:: 3.24

  Finds the NativeHelper library on Android (``libnativehelper.so``), which
  exposes JVM functions such as ``JNI_CreateJavaVM()``.

If no components are specified, the module defaults are:

* When targeting Android with API level 31 and above: module looks for the
  ``NativeHelper`` component.  For other Android API levels, components are by
  default not set.
* When targeting other systems: module looks for ``AWT`` and ``JVM`` components.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``JNI::JNI``
  .. versionadded:: 3.24

  Main target encapsulating all JNI usage requirements, available if ``jni.h``
  is found.

``JNI::AWT``
  .. versionadded:: 3.24

  Target encapsulating the Java AWT Native Interface (JAWT) library usage
  requirements, available if the ``AWT`` component is found.

``JNI::JVM``
  .. versionadded:: 3.24

  Target encapsulating the Java Virtual Machine (JVM) library usage
  requirements, available if component ``JVM`` is found.

``JNI::NativeHelper``
  .. versionadded:: 3.24

  Target encapsulating the NativeHelper library usage requirements, available
  when targeting Android API level 31 and above, and the library is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``JNI_FOUND``
  Boolean indicating whether (the requested version of) JNI was found.

``JNI_<component>_FOUND``
  .. versionadded:: 3.24

  Boolean indicating whether the ``<component>`` was found.

``JNI_VERSION``
  Full Android NDK package version (including suffixes such as ``-beta3`` and
  ``-rc1``) or undefined otherwise.

``JNI_VERSION_MAJOR``
  .. versionadded:: 3.24

  Android NDK major version or undefined otherwise.

``JNI_VERSION_MINOR``
  .. versionadded:: 3.24

  Android NDK minor version or undefined otherwise.

``JNI_VERSION_PATCH``
  .. versionadded:: 3.24

  Android NDK patch version or undefined otherwise.

``JNI_INCLUDE_DIRS``
  The include directories needed to use the JNI.

``JNI_LIBRARIES``
  The libraries (JAWT and JVM) needed to link against to use JNI.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables are also available to set or use:

``JAVA_INCLUDE_PATH``
  The directory containing the ``jni.h`` header.

``JAVA_INCLUDE_PATH2``
  The directory containing machine-dependent headers ``jni_md.h`` and
  ``jniport.h``.  This variable is defined only if ``jni.h`` depends on one of
  these headers.  In contrast, Android NDK ``jni.h`` can be typically used
  standalone.

``JAVA_AWT_INCLUDE_PATH``
  The directory containing the ``jawt.h`` header.

``JAVA_AWT_LIBRARY``
  The path to the Java AWT Native Interface (JAWT) library.

``JAVA_JVM_LIBRARY``
  The path to the Java Virtual Machine (JVM) library.

Hints
^^^^^

This module accepts the following variables:

``JAVA_HOME``
  The caller can set this variable to specify the installation directory of Java
  explicitly.

Examples
^^^^^^^^

Finding JNI and linking it to a project target:

.. code-block:: cmake

  find_package(JNI)
  target_link_libraries(project_target PRIVATE JNI::JNI)

Finding JNI with AWT component specified and linking them to a project target:

.. code-block:: cmake

  find_package(JNI COMPONENTS AWT)
  target_link_libraries(project_target PRIVATE JNI::JNI JNI::AWT)

A more common way to use Java and JNI in CMake is to use a dedicated
:module:`UseJava` module:

.. code-block:: cmake

  find_package(Java)
  find_package(JNI)
  include(UseJava)

See Also
^^^^^^^^

* The :module:`FindJava` module to find Java runtime tools and development
  components.
* The :module:`UseJava` module to use Java in CMake.
#]=======================================================================]

include(CheckSourceCompiles)
include(CMakePushCheckState)
include(FindPackageHandleStandardArgs)

if(NOT JNI_FIND_COMPONENTS)
  if(ANDROID)
    if(CMAKE_SYSTEM_VERSION LESS 31)
      # There are no components for Android NDK
      set(JNI_FIND_COMPONENTS)
    else()
      set(JNI_FIND_COMPONENTS NativeHelper)
      set(JNI_FIND_REQUIRED_NativeHelper TRUE)
    endif()
  else()
    set(JNI_FIND_COMPONENTS AWT JVM)
    # For compatibility purposes, if no components are specified both are
    # considered required.
    set(JNI_FIND_REQUIRED_AWT TRUE)
    set(JNI_FIND_REQUIRED_JVM TRUE)
  endif()
else()
  # On Android, if JVM was requested we need to find NativeHelper as well which
  # is an implicit dependency of JVM allowing to provide uniform access to basic
  # JVM/DVM functionality.
  if(ANDROID AND CMAKE_SYSTEM_VERSION GREATER_EQUAL 31 AND JVM IN_LIST JNI_FIND_COMPONENTS)
    if(NOT NativeHelper IN_LIST JNI_FIND_COMPONENTS)
      list(APPEND JNI_FIND_COMPONENTS NativeHelper)
      # NativeHelper is required only if JVM was requested as such.
      set(JNI_FIND_REQUIRED_NativeHelper ${JNI_FIND_REQUIRED_JVM})
    endif()
  endif()
endif()

# Expand {libarch} occurrences to java_libarch subdirectory(-ies) and set ${_var}
macro(java_append_library_directories _var)
  # Determine java arch-specific library subdir
  # Mostly based on openjdk/jdk/make/common/shared/Platform.gmk as of openjdk
  # 1.6.0_18 + icedtea patches. However, it would be much better to base the
  # guess on the first part of the GNU config.guess platform triplet.
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL "x86_64-linux-gnux32")
      set(_java_libarch "x32" "amd64" "i386")
    else()
      set(_java_libarch "amd64" "i386")
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^i.86$")
    set(_java_libarch "i386")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64")
    set(_java_libarch "arm64" "aarch64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^alpha")
    set(_java_libarch "alpha")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
    # Subdir is "arm" for both big-endian (arm) and little-endian (armel).
    set(_java_libarch "arm" "aarch32")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^mips")
    # mips* machines are bi-endian mostly so processor does not tell
    # endianness of the underlying system.
    set(_java_libarch "${CMAKE_SYSTEM_PROCESSOR}"
        "mips" "mipsel" "mipseb" "mipsr6" "mipsr6el"
        "mips64" "mips64el" "mips64r6" "mips64r6el"
        "mipsn32" "mipsn32el" "mipsn32r6" "mipsn32r6el")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)64le")
    set(_java_libarch "ppc64" "ppc64le")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)64")
    set(_java_libarch "ppc64" "ppc")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)")
    set(_java_libarch "ppc" "ppc64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^sparc")
    # Both flavors can run on the same processor
    set(_java_libarch "${CMAKE_SYSTEM_PROCESSOR}" "sparc" "sparcv9")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(parisc|hppa)")
    set(_java_libarch "parisc" "parisc64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^s390")
    # s390 binaries can run on s390x machines
    set(_java_libarch "${CMAKE_SYSTEM_PROCESSOR}" "s390" "s390x")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^sh")
    set(_java_libarch "sh")
  else()
    set(_java_libarch "${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  # Append default list architectures if CMAKE_SYSTEM_PROCESSOR was empty or
  # system is non-Linux (where the code above has not been well tested)
  if(NOT _java_libarch OR NOT (CMAKE_SYSTEM_NAME MATCHES "Linux"))
    list(APPEND _java_libarch "i386" "amd64" "ppc")
  endif()

  # Sometimes ${CMAKE_SYSTEM_PROCESSOR} is added to the list to prefer
  # current value to a hardcoded list. Remove possible duplicates.
  list(REMOVE_DUPLICATES _java_libarch)

  foreach(_path ${ARGN})
    if(_path MATCHES "{libarch}")
      foreach(_libarch IN LISTS _java_libarch)
        string(REPLACE "{libarch}" "${_libarch}" _newpath "${_path}")
        if(EXISTS ${_newpath})
          list(APPEND ${_var} "${_newpath}")
        endif()
      endforeach()
    else()
      if(EXISTS ${_path})
        list(APPEND ${_var} "${_path}")
      endif()
    endif()
  endforeach()
endmacro()

include(${CMAKE_CURRENT_LIST_DIR}/CMakeFindJavaCommon.cmake)

# Save CMAKE_FIND_FRAMEWORK
if(DEFINED CMAKE_FIND_FRAMEWORK)
  set(_JNI_CMAKE_FIND_FRAMEWORK ${CMAKE_FIND_FRAMEWORK})
else()
  unset(_JNI_CMAKE_FIND_FRAMEWORK)
endif()

if(_JAVA_HOME_EXPLICIT)
  set(CMAKE_FIND_FRAMEWORK NEVER)
endif()

set(JAVA_AWT_LIBRARY_DIRECTORIES)
if(_JAVA_HOME)
  JAVA_APPEND_LIBRARY_DIRECTORIES(JAVA_AWT_LIBRARY_DIRECTORIES
    ${_JAVA_HOME}/jre/lib/{libarch}
    ${_JAVA_HOME}/jre/lib
    ${_JAVA_HOME}/lib/{libarch}
    ${_JAVA_HOME}/lib
    ${_JAVA_HOME}
    )
endif()

if (WIN32)
  set (_JNI_HINTS)
  macro (_JNI_GET_INSTALLED_VERSIONS _KIND)
  cmake_host_system_information(RESULT _JNI_VERSIONS
    QUERY WINDOWS_REGISTRY "HKLM/SOFTWARE/JavaSoft/${_KIND}"
    SUBKEYS)
    if (_JNI_VERSIONS)
      string (REGEX MATCHALL "[0-9._]+" _JNI_VERSIONS "${_JNI_VERSIONS}")
      string (REGEX REPLACE "([0-9._]+)" "\\1" _JNI_VERSIONS "${_JNI_VERSIONS}")
      if (_JNI_VERSIONS)
        # sort versions. Most recent first
        list (SORT _JNI_VERSIONS COMPARE NATURAL ORDER DESCENDING)
        foreach (_JNI_VERSION IN LISTS _JNI_VERSIONS)
          string(REPLACE "_" "." _JNI_CMAKE_VERSION "${_JNI_VERSION}")
          if (JNI_FIND_VERSION_EXACT
              AND NOT _JNI_CMAKE_VERSION MATCHES "^${JNI_FIND_VERSION}")
            continue()
          endif()
          if (DEFINED JNI_FIND_VERSION AND _JNI_CMAKE_VERSION VERSION_LESS JNI_FIND_VERSION)
            break()
          endif()
          list(APPEND _JNI_HINTS "[HKEY_LOCAL_MACHINE\\SOFTWARE\\JavaSoft\\${_KIND}\\${_JNI_VERSION};JavaHome]")
        endforeach()
      endif()
    endif()
  endmacro()

    # for version 9 and upper
  _JNI_GET_INSTALLED_VERSIONS("JDK")

  # for versions older than 9
  _JNI_GET_INSTALLED_VERSIONS("Java Development Kit")

  foreach (_JNI_HINT IN LISTS _JNI_HINTS)
    list(APPEND JAVA_AWT_LIBRARY_DIRECTORIES "${_JNI_HINT}/lib")
  endforeach()
endif()

set(_JNI_JAVA_DIRECTORIES_BASE
  /usr/lib/jvm/java
  /usr/lib/java
  /usr/lib/jvm
  /usr/local/lib/java
  /usr/local/share/java
  /usr/lib/j2sdk1.4-sun
  /usr/lib/j2sdk1.5-sun
  /opt/sun-jdk-1.5.0.04
  /usr/lib/jvm/java-6-sun
  /usr/lib/jvm/java-1.5.0-sun
  /usr/lib/jvm/java-6-sun-1.6.0.00       # can this one be removed according to #8821 ? Alex
  /usr/lib/jvm/java-6-openjdk
  /usr/lib/jvm/java-1.6.0-openjdk-1.6.0.0        # fedora
  # Debian specific paths for default JVM
  /usr/lib/jvm/default-java
  # Arch Linux specific paths for default JVM
  /usr/lib/jvm/default
  # Ubuntu specific paths for default JVM
  /usr/lib/jvm/java-26-openjdk-{libarch}
  /usr/lib/jvm/java-25-openjdk-{libarch}    # Ubuntu 24.04 LTS
  /usr/lib/jvm/java-24-openjdk-{libarch}
  /usr/lib/jvm/java-23-openjdk-{libarch}
  /usr/lib/jvm/java-22-openjdk-{libarch}
  /usr/lib/jvm/java-21-openjdk-{libarch}    # Ubuntu 23.04
  /usr/lib/jvm/java-20-openjdk-{libarch}    # Ubuntu 22.10
  /usr/lib/jvm/java-19-openjdk-{libarch}    # Ubuntu 22.04 LTS
  /usr/lib/jvm/java-18-openjdk-{libarch}    # Ubuntu 22.04 LTS
  /usr/lib/jvm/java-17-openjdk-{libarch}    # Ubuntu 18.04 LTS
  /usr/lib/jvm/java-16-openjdk-{libarch}    # Ubuntu 20.04 LTS
  /usr/lib/jvm/java-13-openjdk-{libarch}    # Ubuntu 20.04 LTS
  /usr/lib/jvm/java-11-openjdk-{libarch}    # Ubuntu 18.04 LTS
  /usr/lib/jvm/java-8-openjdk-{libarch}     # Ubuntu 15.10
  /usr/lib/jvm/java-7-openjdk-{libarch}     # Ubuntu 15.10
  /usr/lib/jvm/java-6-openjdk-{libarch}     # Ubuntu 15.10
  # OpenBSD specific paths for default JVM
  /usr/local/jdk-1.7.0
  /usr/local/jre-1.7.0
  /usr/local/jdk-1.6.0
  /usr/local/jre-1.6.0
  # FreeBSD specific paths for default JVM
  /usr/local/openjdk15
  /usr/local/openjdk14
  /usr/local/openjdk13
  /usr/local/openjdk12
  /usr/local/openjdk11
  /usr/local/openjdk8
  /usr/local/openjdk7
  # SuSE specific paths for default JVM
  /usr/lib64/jvm/java
  /usr/lib64/jvm/jre
  )

set(_JNI_JAVA_AWT_LIBRARY_TRIES)
set(_JNI_JAVA_INCLUDE_TRIES)

foreach(_java_dir IN LISTS _JNI_JAVA_DIRECTORIES_BASE)
  list(APPEND _JNI_JAVA_AWT_LIBRARY_TRIES
    ${_java_dir}/jre/lib/{libarch}
    ${_java_dir}/jre/lib
    ${_java_dir}/lib/{libarch}
    ${_java_dir}/lib
    ${_java_dir}
  )
  list(APPEND _JNI_JAVA_INCLUDE_TRIES
    ${_java_dir}/include
  )
endforeach()

JAVA_APPEND_LIBRARY_DIRECTORIES(JAVA_AWT_LIBRARY_DIRECTORIES
    ${_JNI_JAVA_AWT_LIBRARY_TRIES}
  )

set(JAVA_JVM_LIBRARY_DIRECTORIES)
foreach(dir IN LISTS JAVA_AWT_LIBRARY_DIRECTORIES)
  list(APPEND JAVA_JVM_LIBRARY_DIRECTORIES
    "${dir}"
    "${dir}/client"
    "${dir}/minimal"
    "${dir}/server"
    "${dir}/zero"
    # IBM SDK, Java Technology Edition, specific paths
    "${dir}/j9vm"
    "${dir}/default"
    )
endforeach()

set(JAVA_AWT_INCLUDE_DIRECTORIES)
if(_JAVA_HOME)
  list(APPEND JAVA_AWT_INCLUDE_DIRECTORIES ${_JAVA_HOME}/include)
endif()
if (WIN32)
  foreach (_JNI_HINT IN LISTS _JNI_HINTS)
    list(APPEND JAVA_AWT_INCLUDE_DIRECTORIES "${_JNI_HINT}/include")
  endforeach()
endif()

JAVA_APPEND_LIBRARY_DIRECTORIES(JAVA_AWT_INCLUDE_DIRECTORIES
  ${_JNI_JAVA_INCLUDE_TRIES}
  )

foreach(JAVA_PROG IN ITEMS "${JAVA_RUNTIME}" "${JAVA_COMPILE}" "${JAVA_ARCHIVE}")
  get_filename_component(jpath "${JAVA_PROG}" PATH)
  foreach(JAVA_INC_PATH IN ITEMS ../include ../java/include ../share/java/include)
    if(EXISTS ${jpath}/${JAVA_INC_PATH})
      list(APPEND JAVA_AWT_INCLUDE_DIRECTORIES "${jpath}/${JAVA_INC_PATH}")
    endif()
  endforeach()
  foreach(JAVA_LIB_PATH IN ITEMS
    ../lib ../jre/lib ../jre/lib/i386
    ../java/lib ../java/jre/lib ../java/jre/lib/i386
    ../share/java/lib ../share/java/jre/lib ../share/java/jre/lib/i386)
    if(EXISTS ${jpath}/${JAVA_LIB_PATH})
      list(APPEND JAVA_AWT_LIBRARY_DIRECTORIES "${jpath}/${JAVA_LIB_PATH}")
    endif()
  endforeach()
endforeach()

if(APPLE)
  if(DEFINED XCODE_VERSION)
    set(_FindJNI_XCODE_VERSION "${XCODE_VERSION}")
  else()
    # get xcode version
    execute_process(
      COMMAND xcodebuild -version
      OUTPUT_VARIABLE _FindJNI_XCODEBUILD_VERSION
      ERROR_VARIABLE _FindJNI_XCODEBUILD_VERSION
      RESULT_VARIABLE _FindJNI_XCODEBUILD_RESULT
      )
    if(_FindJNI_XCODEBUILD_RESULT EQUAL 0 AND _FindJNI_XCODEBUILD_VERSION MATCHES "Xcode ([0-9]+(\\.[0-9]+)*)")
      set(_FindJNI_XCODE_VERSION "${CMAKE_MATCH_1}")
    else()
      set(_FindJNI_XCODE_VERSION "")
    endif()
    unset(_FindJNI_XCODEBUILD_VERSION)
  endif()

  if(_FindJNI_XCODE_VERSION VERSION_GREATER 12.1)
    set(CMAKE_FIND_FRAMEWORK "NEVER")
  endif()
  unset(_FindJNI_XCODE_VERSION)

  if(CMAKE_FIND_FRAMEWORK STREQUAL "ONLY")
    set(_JNI_SEARCHES FRAMEWORK)
  elseif(CMAKE_FIND_FRAMEWORK STREQUAL "NEVER")
    set(_JNI_SEARCHES NORMAL)
  elseif(CMAKE_FIND_FRAMEWORK STREQUAL "LAST")
    set(_JNI_SEARCHES NORMAL FRAMEWORK)
  else()
    set(_JNI_SEARCHES FRAMEWORK NORMAL)
  endif()
  set(_JNI_FRAMEWORK_JVM NAMES JavaVM)
  set(_JNI_FRAMEWORK_JAWT "${_JNI_FRAMEWORK_JVM}")
else()
  set(_JNI_SEARCHES NORMAL)
endif()

set(_JNI_NORMAL_JVM
  NAMES jvm
  PATHS ${JAVA_JVM_LIBRARY_DIRECTORIES}
  )

set(_JNI_NORMAL_JAWT
  NAMES jawt
  PATHS ${JAVA_AWT_LIBRARY_DIRECTORIES}
  )

foreach(search IN LISTS _JNI_SEARCHES)
  if(JVM IN_LIST JNI_FIND_COMPONENTS)
    find_library(JAVA_JVM_LIBRARY ${_JNI_${search}_JVM}
      DOC "Java Virtual Machine library"
    )
  endif()

  if(AWT IN_LIST JNI_FIND_COMPONENTS)
    find_library(JAVA_AWT_LIBRARY ${_JNI_${search}_JAWT}
      DOC "Java AWT Native Interface library"
    )
    if(JAVA_JVM_LIBRARY)
      break()
    endif()
  endif()
endforeach()
unset(_JNI_SEARCHES)
unset(_JNI_FRAMEWORK_JVM)
unset(_JNI_FRAMEWORK_JAWT)
unset(_JNI_NORMAL_JVM)
unset(_JNI_NORMAL_JAWT)

# Find headers matching the library.
if("${JAVA_JVM_LIBRARY};${JAVA_AWT_LIBRARY};" MATCHES "(/JavaVM.framework|-framework JavaVM);")
  set(CMAKE_FIND_FRAMEWORK ONLY)
else()
  set(CMAKE_FIND_FRAMEWORK NEVER)
endif()

# add in the include path
find_path(JAVA_INCLUDE_PATH jni.h
  ${JAVA_AWT_INCLUDE_DIRECTORIES}
  DOC "JNI include directory"
)

if(JAVA_INCLUDE_PATH)
  if(CMAKE_C_COMPILER_LOADED)
    set(_JNI_CHECK_LANG C)
  elseif(CMAKE_CXX_COMPILER_LOADED)
    set(_JNI_CHECK_LANG CXX)
  else()
    set(_JNI_CHECK_LANG FALSE)
  endif()

  # Skip the check if neither C nor CXX is loaded.
  if(_JNI_CHECK_LANG)
    cmake_push_check_state(RESET)
    # The result of the following check is not relevant for the user as
    # JAVA_INCLUDE_PATH2 will be added to REQUIRED_VARS if necessary.
    set(CMAKE_REQUIRED_QUIET ON)
    set(CMAKE_REQUIRED_INCLUDES ${JAVA_INCLUDE_PATH})

    # Determine whether jni.h requires jni_md.h and add JAVA_INCLUDE_PATH2
    # correspondingly to REQUIRED_VARS
    check_source_compiles(${_JNI_CHECK_LANG}
"
#include <jni.h>
int main(void) { return 0; }
"
      JNI_INCLUDE_PATH2_OPTIONAL)

    cmake_pop_check_state()
  else()
    # If the above check is skipped assume jni_md.h is not needed.
    set(JNI_INCLUDE_PATH2_OPTIONAL TRUE)
  endif()

  unset(_JNI_CHECK_LANG)
endif()

find_path(JAVA_INCLUDE_PATH2 NAMES jni_md.h jniport.h
  PATHS ${JAVA_INCLUDE_PATH}
  ${JAVA_INCLUDE_PATH}/darwin
  ${JAVA_INCLUDE_PATH}/win32
  ${JAVA_INCLUDE_PATH}/linux
  ${JAVA_INCLUDE_PATH}/freebsd
  ${JAVA_INCLUDE_PATH}/openbsd
  ${JAVA_INCLUDE_PATH}/solaris
  ${JAVA_INCLUDE_PATH}/hp-ux
  ${JAVA_INCLUDE_PATH}/alpha
  ${JAVA_INCLUDE_PATH}/aix
  DOC "jni_md.h jniport.h include directory"
)

if(AWT IN_LIST JNI_FIND_COMPONENTS)
  find_path(JAVA_AWT_INCLUDE_PATH jawt.h
    ${JAVA_INCLUDE_PATH}
    DOC "Java AWT Native Interface include directory"
  )
endif()

if(ANDROID)
  # Some functions in jni.h (e.g., JNI_GetCreatedJavaVMs) are exported by
  # libnativehelper.so, however, only when targeting Android API level >= 31.
  find_library(JAVA_NativeHelper_LIBRARY NAMES nativehelper
    DOC "Android nativehelper library"
  )
endif()

# Set found components
if(JAVA_AWT_INCLUDE_PATH AND JAVA_AWT_LIBRARY)
  set(JNI_AWT_FOUND TRUE)
else()
  set(JNI_AWT_FOUND FALSE)
endif()

# JVM is available even on Android referencing the nativehelper library
if(JAVA_JVM_LIBRARY)
  set(JNI_JVM_FOUND TRUE)
else()
  set(JNI_JVM_FOUND FALSE)
endif()

if(JAVA_NativeHelper_LIBRARY)
  # Alias JAVA_JVM_LIBRARY to JAVA_NativeHelper_LIBRARY
  if(NOT JAVA_JVM_LIBRARY)
    set(JAVA_JVM_LIBRARY "${JAVA_NativeHelper_LIBRARY}" CACHE FILEPATH
      "Alias to nativehelper library" FORCE)
    # Make JVM component available
    set(JNI_JVM_FOUND TRUE)
  endif()
  set(JNI_NativeHelper_FOUND TRUE)
else()
  set(JNI_NativeHelper_FOUND FALSE)
endif()

# Restore CMAKE_FIND_FRAMEWORK
if(DEFINED _JNI_CMAKE_FIND_FRAMEWORK)
  set(CMAKE_FIND_FRAMEWORK ${_JNI_CMAKE_FIND_FRAMEWORK})
  unset(_JNI_CMAKE_FIND_FRAMEWORK)
else()
  unset(CMAKE_FIND_FRAMEWORK)
endif()

if(ANDROID)
  # Extract NDK version from source.properties in the NDK root
  set(JAVA_SOURCE_PROPERTIES_FILE ${CMAKE_ANDROID_NDK}/source.properties)

  if(EXISTS ${JAVA_SOURCE_PROPERTIES_FILE})
    file(READ ${JAVA_SOURCE_PROPERTIES_FILE} NDK_VERSION_CONTENTS)
    string (REGEX REPLACE
      ".*Pkg\\.Revision = (([0-9]+)\\.([0-9]+)\\.([0-9]+)([^\n]+)?).*" "\\1"
      JNI_VERSION "${NDK_VERSION_CONTENTS}")
    set(JNI_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(JNI_VERSION_MINOR ${CMAKE_MATCH_2})
    set(JNI_VERSION_PATCH ${CMAKE_MATCH_3})
    set(JNI_VERSION_COMPONENTS 3)

    set(JNI_FPHSA_ARGS VERSION_VAR JNI_VERSION HANDLE_VERSION_RANGE)
  endif()
endif()

set(JNI_REQUIRED_VARS JAVA_INCLUDE_PATH)

if(NOT JNI_INCLUDE_PATH2_OPTIONAL)
  list(APPEND JNI_REQUIRED_VARS JAVA_INCLUDE_PATH2)
endif()

find_package_handle_standard_args(JNI
  REQUIRED_VARS ${JNI_REQUIRED_VARS}
  ${JNI_FPHSA_ARGS}
  HANDLE_COMPONENTS
)

mark_as_advanced(
  JAVA_AWT_LIBRARY
  JAVA_JVM_LIBRARY
  JAVA_AWT_INCLUDE_PATH
  JAVA_INCLUDE_PATH
  JAVA_INCLUDE_PATH2
)

set(JNI_LIBRARIES)

foreach(component IN LISTS JNI_FIND_COMPONENTS)
  if(JNI_${component}_FOUND)
    list(APPEND JNI_LIBRARIES ${JAVA_${component}_LIBRARY})
  endif()
endforeach()

set(JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH})

if(NOT JNI_INCLUDE_PATH2_OPTIONAL)
  list(APPEND JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH2})
endif()

if(JNI_FIND_REQUIRED_AWT)
  list(APPEND JNI_INCLUDE_DIRS ${JAVA_AWT_INCLUDE_PATH})
endif()

if(JNI_FOUND)
  if(NOT TARGET JNI::JNI)
    add_library(JNI::JNI IMPORTED INTERFACE)
  endif()

  set_property(TARGET JNI::JNI PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${JAVA_INCLUDE_PATH})

  if(JNI_NativeHelper_FOUND)
    if(NOT TARGET JNI::NativeHelper)
      add_library(JNI::NativeHelper IMPORTED UNKNOWN)
    endif()

    set_property(TARGET JNI::NativeHelper PROPERTY INTERFACE_LINK_LIBRARIES
      JNI::JNI)
    set_property(TARGET JNI::NativeHelper PROPERTY IMPORTED_LOCATION
      ${JAVA_NativeHelper_LIBRARY})
  endif()

  if(NOT JNI_INCLUDE_PATH2_OPTIONAL AND JAVA_INCLUDE_PATH2)
    set_property(TARGET JNI::JNI APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${JAVA_INCLUDE_PATH2})
  endif()

  if(JNI_AWT_FOUND)
    if(NOT TARGET JNI::AWT)
      add_library(JNI::AWT IMPORTED UNKNOWN)
    endif()

    set_property(TARGET JNI::AWT PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${JAVA_AWT_INCLUDE_PATH})
    set_property(TARGET JNI::AWT PROPERTY IMPORTED_LOCATION
      ${JAVA_AWT_LIBRARY})
    set_property(TARGET JNI::AWT PROPERTY INTERFACE_LINK_LIBRARIES JNI::JNI)
  endif()

  if(JNI_JVM_FOUND OR JNI_NativeHelper_FOUND)
    # If Android nativehelper is available but not the JVM library, we still
    # define the JNI::JVM target but only declare JNI::NativeHelper as an
    # interface link library of the former. This provides a uniform access to
    # fundamental JVM functionality regardless of whether JVM or DVM is used. At
    # the same time, this allows the user to detect whenever exclusively
    # nativehelper functionality is available.
    if(NOT TARGET JNI::JVM)
      if(JAVA_JVM_LIBRARY AND NOT JAVA_JVM_LIBRARY STREQUAL JAVA_NativeHelper_LIBRARY)
        # JAVA_JVM_LIBRARY is not an alias of JAVA_NativeHelper_LIBRARY
        add_library(JNI::JVM IMPORTED UNKNOWN)
      else()
        add_library(JNI::JVM IMPORTED INTERFACE)
      endif()
    endif()

    set_property(TARGET JNI::JVM PROPERTY INTERFACE_LINK_LIBRARIES JNI::JNI)
    get_property(_JNI_JVM_TYPE TARGET JNI::JVM PROPERTY TYPE)

    if(NOT _JNI_JVM_TYPE STREQUAL "INTERFACE_LIBRARY")
      set_property(TARGET JNI::JVM PROPERTY IMPORTED_LOCATION
        ${JAVA_JVM_LIBRARY})
    else()
      # We declare JNI::NativeHelper a dependency of JNI::JVM only if the latter
      # was not initially found. If the solely theoretical situation occurs
      # where both libraries are available, we want to avoid any potential
      # errors that can occur due to duplicate symbols.
      set_property(TARGET JNI::JVM APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        JNI::NativeHelper)
    endif()

    unset(_JNI_JVM_TYPE)
  endif()
endif()
