# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
UseEcos
-------

This module defines variables and provides commands required to build an eCos
application.

Load this module in a CMake project with:

.. code-block:: cmake

  include(UseEcos)

Commands
^^^^^^^^

This module provides the following commands:

Building an eCos Application
""""""""""""""""""""""""""""

.. command:: ecos_add_include_directories

  Adds the eCos include directories for the current ``CMakeLists.txt`` file:

  .. code-block:: cmake

    ecos_add_include_directories()

.. command:: ecos_adjust_directory

  Adjusts the paths of given source files:

  .. code-block:: cmake

    ecos_adjust_directory(<var> <sources>...)

  This command modifies the paths of the source files ``<sources>...`` to
  make them suitable for use with ``ecos_add_executable()``, and stores them
  in the variable ``<var>``.

  ``<var>``
    Result variable name holding a new list of source files with adjusted paths.
  ``<sources>...``
    A list of relative or absolute source files to adjust their paths.

  Use this command when the actual sources are located one level upwards. A
  ``../`` has to be prepended in front of every source file that is given as a
  relative path.

.. command:: ecos_add_executable

  Creates an eCos application executable:

  .. code-block:: cmake

    ecos_add_executable(<name> <sources>...)

  ``<name>``
    The name of the executable.
  ``<sources>...``
    A list of all source files, where the path has been adjusted beforehand by
    calling the ``ecos_adjust_directory()``.

  This command also sets the ``ECOS_DEFINITIONS`` local variable, holding some
  common compile definitions.

Selecting the Toolchain
"""""""""""""""""""""""

.. command:: ecos_use_arm_elf_tools

  Enables the ARM ELF toolchain for the directory where it is called:

  .. code-block:: cmake

    ecos_use_arm_elf_tools()

  Use this command, when compiling for the xscale processor.

.. command:: ecos_use_i386_elf_tools

  Enables the i386 ELF toolchain for the directory where it is called:

  .. code-block:: cmake

    ecos_use_i386_elf_tools()

.. command:: ecos_use_ppc_eabi_tools

  Enables the PowerPC toolchain for the directory where it is called:

  .. code-block:: cmake

    ecos_use_ppc_eabi_tools()

Variables
^^^^^^^^^

This module also defines the following variables:

``ECOSCONFIG_EXECUTABLE``
  Cache variable that contains a path to the ``ecosconfig`` executable (the eCos
  configuration program).

``ECOS_CONFIG_FILE``
  A local variable that defaults to ``ecos.ecc``.  If eCos configuration file
  has a different name, adjust this variable before calling the
  ``ecos_add_executable()``.

Examples
^^^^^^^^

The following example demonstrates defining an eCos executable target in a
project that follows the common eCos convention of listing source files in
a ``ProjectSources.txt`` file, located one directory above the current
``CMakeLists.txt``:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(UseEcos)

  # Add the eCos include directories.
  ecos_add_include_directories()

  # Include the file with the eCos sources list. This file, for example, defines
  # a list of eCos sources:
  #   set(sources file_1.cxx file_2.cxx file_3.cxx)
  include(../ProjectSources.txt)

  # When using such directory structure, relative source paths must be adjusted:
  ecos_adjust_directory(adjusted_sources ${sources})

  # Create eCos executable.
  ecos_add_executable(ecos_app ${adjusted_sources})
#]=======================================================================]

# First check that ecosconfig is available.
find_program(ECOSCONFIG_EXECUTABLE NAMES ecosconfig)
mark_as_advanced(ECOSCONFIG_EXECUTABLE)
if(NOT ECOSCONFIG_EXECUTABLE)
  message(SEND_ERROR "ecosconfig was not found. Either include it in the system path or set it manually using ccmake.")
else()
  message(STATUS "Found ecosconfig: ${ECOSCONFIG_EXECUTABLE}")
endif()

# Check that ECOS_REPOSITORY is set correctly.
if (NOT EXISTS $ENV{ECOS_REPOSITORY}/ecos.db)
  message(SEND_ERROR "The environment variable ECOS_REPOSITORY is not set correctly. Set it to the directory which contains the file ecos.db")
else ()
  message(STATUS "ECOS_REPOSITORY is set to $ENV{ECOS_REPOSITORY}")
endif ()

# Check that tclsh (coming with TCL) is available, otherwise ecosconfig doesn't
# work.
find_package(Tclsh)
if (NOT Tclsh_FOUND)
  message(SEND_ERROR "The TCL tclsh was not found. Please install TCL, it is required for building eCos applications.")
endif ()

macro(ECOS_ADD_INCLUDE_DIRECTORIES)
  # Check for ProjectSources.txt one level higher.
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../ProjectSources.txt)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
  else ()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
  endif ()

  # The ecos include directory.
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/ecos/install/include/)
endmacro()

macro (ECOS_USE_ARM_ELF_TOOLS)
  set(CMAKE_CXX_COMPILER "arm-elf-c++")
  set(CMAKE_COMPILER_IS_GNUCXX 1)
  set(CMAKE_C_COMPILER "arm-elf-gcc")
  set(CMAKE_AR "arm-elf-ar")
  set(CMAKE_RANLIB "arm-elf-ranlib")
  # For linking.
  set(ECOS_LD_MCPU "-mcpu=xscale")
  # For compiling.
  add_definitions(-mcpu=xscale -mapcs-frame)
  # For the obj-tools.
  set(ECOS_ARCH_PREFIX "arm-elf-")
endmacro ()

macro (ECOS_USE_PPC_EABI_TOOLS)
  set(CMAKE_CXX_COMPILER "powerpc-eabi-c++")
  set(CMAKE_COMPILER_IS_GNUCXX 1)
  set(CMAKE_C_COMPILER "powerpc-eabi-gcc")
  set(CMAKE_AR "powerpc-eabi-ar")
  set(CMAKE_RANLIB "powerpc-eabi-ranlib")
  # For linking.
  set(ECOS_LD_MCPU "")
  # For compiling.
  add_definitions()
  # For the obj-tools.
  set(ECOS_ARCH_PREFIX "powerpc-eabi-")
endmacro ()

macro (ECOS_USE_I386_ELF_TOOLS)
  set(CMAKE_CXX_COMPILER "i386-elf-c++")
  set(CMAKE_COMPILER_IS_GNUCXX 1)
  set(CMAKE_C_COMPILER "i386-elf-gcc")
  set(CMAKE_AR "i386-elf-ar")
  set(CMAKE_RANLIB "i386-elf-ranlib")
  # For linking.
  set(ECOS_LD_MCPU "")
  # For compiling.
  add_definitions()
  # For the obj-tools.
  set(ECOS_ARCH_PREFIX "i386-elf-")
endmacro ()

macro(ECOS_ADJUST_DIRECTORY _target_FILES )
  foreach (_current_FILE ${ARGN})
    get_filename_component(_abs_FILE ${_current_FILE} ABSOLUTE)
    if (NOT ${_abs_FILE} STREQUAL ${_current_FILE})
      get_filename_component(_abs_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../${_current_FILE} ABSOLUTE)
    endif ()
    list(APPEND ${_target_FILES} ${_abs_FILE})
  endforeach ()
endmacro()

# The default eCos config file name. Maybe in future also out-of-source builds
# may be possible.
set(ECOS_CONFIG_FILE ecos.ecc)

# Internal macro that creates the dependency from all source files on the eCos
# target.ld and adds the command for compiling eCos.
macro(ECOS_ADD_TARGET_LIB)
  # When building out-of-source, create the ecos/ subdir.
  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/ecos)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ecos)
  endif()

  # Sources depend on target.ld.
  set_source_files_properties(
    ${ARGN}
    PROPERTIES
    OBJECT_DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/ecos/install/lib/target.ld
  )

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ecos/install/lib/target.ld
    COMMAND sh -c \"make -C ${CMAKE_CURRENT_BINARY_DIR}/ecos || exit -1\; if [ -e ${CMAKE_CURRENT_BINARY_DIR}/ecos/install/lib/target.ld ] \; then touch ${CMAKE_CURRENT_BINARY_DIR}/ecos/install/lib/target.ld\; fi\"
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ecos/makefile
  )

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ecos/makefile
    COMMAND sh -c \" cd ${CMAKE_CURRENT_BINARY_DIR}/ecos\; ${ECOSCONFIG_EXECUTABLE} --config=${CMAKE_CURRENT_SOURCE_DIR}/ecos/${ECOS_CONFIG_FILE} tree || exit -1\;\"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ecos/${ECOS_CONFIG_FILE}
  )

  add_custom_target( ecos make -C ${CMAKE_CURRENT_BINARY_DIR}/ecos/ DEPENDS  ${CMAKE_CURRENT_BINARY_DIR}/ecos/makefile )
endmacro()

# Get the directory of the current file, used later on in the file.
get_filename_component( ECOS_CMAKE_MODULE_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)

macro(ECOS_ADD_EXECUTABLE _exe_NAME )
  # Definitions, valid for all eCos projects.
  # The optimization and "-g" for debugging has to be enabled in the
  # project-specific CMakeLists.txt.
  add_definitions(-D__ECOS__=1 -D__ECOS=1)
  set(ECOS_DEFINITIONS -Wall -Wno-long-long -pipe -fno-builtin)

  # The executable depends on eCos target.ld.
  ecos_add_target_lib(${ARGN})

  # When using nmake makefiles, the custom buildtype suppresses the default
  # cl.exe flags and the rules for creating objects are adjusted for gcc.
  set(CMAKE_BUILD_TYPE CUSTOM_ECOS_BUILD)
  set(CMAKE_C_COMPILE_OBJECT     "<CMAKE_C_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
  set(CMAKE_CXX_COMPILE_OBJECT   "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")

  # Special link commands for eCos executables.
  set(CMAKE_CXX_LINK_EXECUTABLE  "<CMAKE_CXX_COMPILER> <LINK_FLAGS> <OBJECTS> -o <TARGET> ${_ecos_EXTRA_LIBS} -nostdlib -nostartfiles -L${CMAKE_CURRENT_BINARY_DIR}/ecos/install/lib -Ttarget.ld ${ECOS_LD_MCPU}")
  set(CMAKE_C_LINK_EXECUTABLE    "<CMAKE_C_COMPILER> <LINK_FLAGS> <OBJECTS> -o <TARGET> ${_ecos_EXTRA_LIBS} -nostdlib -nostartfiles -L${CMAKE_CURRENT_BINARY_DIR}/ecos/install/lib -Ttarget.ld ${ECOS_LD_MCPU}")

  # Some strict compiler flags.
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstrict-prototypes")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual -fno-rtti -Wctor-dtor-privacy -fno-strict-aliasing -fno-exceptions")

  add_executable(${_exe_NAME} ${ARGN})
  set_target_properties(${_exe_NAME} PROPERTIES SUFFIX ".elf")

  # Create a binary file.
  add_custom_command(
    TARGET ${_exe_NAME}
    POST_BUILD
    COMMAND ${ECOS_ARCH_PREFIX}objcopy
    ARGS -O binary ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.elf ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.bin
  )

  # And an srec file.
  add_custom_command(
    TARGET ${_exe_NAME}
    POST_BUILD
    COMMAND ${ECOS_ARCH_PREFIX}objcopy
    ARGS -O srec ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.elf ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.srec
  )

  # Add the created files to the clean-files.
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.bin"
    "${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.srec"
    "${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.lst")

  add_custom_target(ecosclean ${CMAKE_COMMAND} -DECOS_DIR=${CMAKE_CURRENT_BINARY_DIR}/ecos/ -P ${ECOS_CMAKE_MODULE_DIR}/ecos_clean.cmake  )
  add_custom_target(normalclean ${CMAKE_MAKE_PROGRAM} clean WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies (ecosclean normalclean)

  add_custom_target( listing
    COMMAND echo -e   \"\\n--- Symbols sorted by address ---\\n\" > ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.lst
    COMMAND ${ECOS_ARCH_PREFIX}nm -S -C -n ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.elf >> ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.lst
    COMMAND echo -e \"\\n--- Symbols sorted by size ---\\n\" >> ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.lst
    COMMAND ${ECOS_ARCH_PREFIX}nm -S -C -r --size-sort ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.elf >> ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.lst
    COMMAND echo -e \"\\n--- Full assembly listing ---\\n\" >> ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.lst
    COMMAND ${ECOS_ARCH_PREFIX}objdump -S -x -d -C ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.elf >> ${CMAKE_CURRENT_BINARY_DIR}/${_exe_NAME}.lst )
endmacro()
