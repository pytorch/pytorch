# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[======================================================================[.rst:
AndroidTestUtilities
------------------------

.. versionadded:: 3.7

This module provides a command to create a test that pushes data needed for
testing an Android device behavior onto a connected Android device.

Load this module in a CMake project with:

.. code-block:: cmake

  include(AndroidTestUtilities)

Commands
^^^^^^^^

This module provides the following command:

.. command:: android_add_test_data

  Creates a test that automatically loads specified data onto an Android
  device:

  .. code-block:: cmake

    android_add_test_data(
      <test-name>
      [FILES <files>...]
      [FILES_DEST <device-dir>]
      [LIBS <libs>...]
      [LIBS_DEST <device-dir>]
      DEVICE_OBJECT_STORE <device-dir>
      DEVICE_TEST_DIR <device-dir>
      [NO_LINK_REGEX <regexes>...]
    )

  This command accepts files and libraries needed to run project-specific
  tests as well as separate destinations for each.  It will create a test
  that loads the files into a device object store and link to them from the
  specified destination.  The files are only uploaded if they are not
  already in the object store.

  On the host operating system, files and libraries are copied at build
  time.  For on-device testing, the files are loaded onto the device by the
  manufactured test at run time.

  This command accepts the following named parameters:

  ``FILES <files>...``
    Zero or more files needed for testing.
  ``FILES_DEST <device-dir>``
    Absolute path where the data files are expected to be.
  ``LIBS <libs>...``
    Zero or more libraries needed for testing.
  ``LIBS_DEST <device-dir>``
    Absolute path where the libraries are expected to be.
  ``DEVICE_OBJECT_STORE <device-dir>``
    Absolute path to the on-device location where the data files are initially
    stored.
  ``DEVICE_TEST_DIR <device-dir>``
    Absolute path to the root directory of the on-device test location.
  ``NO_LINK_REGEX <regexes>...``
    A list of regular expression patterns matching file names to be copied
    from the object store to the test directory, instead of being symlinked.

Examples
^^^^^^^^

The following example shows how to use this module to create a test named
``example_setup_test`` that prepares data during the build phase.  This test
can then be run using :manual:`ctest(1)` to load the data onto the Android
device.

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(AndroidTestUtilities)

  android_add_test_data(
    example_setup_test
    FILES data/protobuffer.p data/file.txt
    LIBS libs/library_1 libs/library_2
    DEVICE_OBJECT_STORE "/sdcard/.ExternalData/SHA"
    DEVICE_TEST_DIR "/data/local/tests/example"
  )
#]======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/ExternalData.cmake)

# The parameters to this function should be set to the list of directories,
# files, and libraries that need to be installed prior to testing.
function(android_add_test_data test_name)
  # As the names suggest, oneValueArgs lists the arguments that specify a
  # single value, while multiValueArgs can contain one or more values.
  set(keywordArgs)
  set(oneValueArgs FILES_DEST LIBS_DEST DEVICE_OBJECT_STORE DEVICE_TEST_DIR)
  set(multiValueArgs FILES LIBS NO_LINK_REGEX)

  # For example, if you called this function with FILES </path/to/file>
  # then this path would be stored in the variable AST_FILES.
  # The AST prefix stands for the name of this function (android_add_test_data).
  cmake_parse_arguments(AST "${keywordArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT AST_DEVICE_TEST_DIR)
    message(FATAL_ERROR "-- You must specify the location of the on device test directory.")
  endif()
  if(NOT AST_DEVICE_OBJECT_STORE)
    message(FATAL_ERROR "-- You must specify the location of the on device object store.")
  endif()
  if(${AST_DEVICE_TEST_DIR} STREQUAL "/")
    message(FATAL_ERROR "-- The device test directory cannot be '/'")
  endif()

  # Copy all test data files into the binary directory, where tests are run.
  # ExternalData will handle fetching DATA{...} references.
  string(REPLACE "|" ";" hash_algs "${_ExternalData_REGEX_EXT}")
  # Convert ExternalData placeholder file names to DATA{} syntax.
  foreach(alg ${hash_algs})
    string(REGEX REPLACE "([^ ;]+)\\.${alg}" "DATA{\\1}" AST_FILES "${AST_FILES}")
  endforeach()

  set(DATA_TARGET_NAME "${test_name}")
  string(FIND "${AST_FILES}" "DATA{" data_files_found)
  if(${data_files_found} GREATER "-1")
    # Use ExternalData if any DATA{} files were found.
    ExternalData_Expand_Arguments(
      ${DATA_TARGET_NAME}
      extern_data_output
      ${AST_FILES})
    ExternalData_Add_Target(${DATA_TARGET_NAME})
  else()
    add_custom_target(${DATA_TARGET_NAME} ALL)
    set(extern_data_output ${AST_FILES})
  endif()

  # For regular files on Linux, just copy them directly.
  foreach(path ${AST_FILES})
    foreach(output ${extern_data_output})
      if(${output} STREQUAL ${path})
        # Check if a destination was specified.  If not, we copy by default
        # into this project's binary directory, preserving its relative path.
        if(AST_${VAR}_DEST)
          set(DEST ${CMAKE_BINARY_DIR}/${parent_dir}/${AST_${VAR}_DEST})
        else()
          get_filename_component(parent_dir ${path} DIRECTORY)
          set(DEST "${CMAKE_BINARY_DIR}/${parent_dir}")
        endif()
        get_filename_component(extern_data_source ${output} REALPATH)
        get_filename_component(extern_data_basename ${output} NAME)
        add_custom_command(
          TARGET ${DATA_TARGET_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${extern_data_source} ${DEST}/${extern_data_basename}
        )
      endif()
    endforeach()
  endforeach()

  if(ANDROID)
    string(REGEX REPLACE "DATA{([^ ;]+)}" "\\1"  processed_FILES "${AST_FILES}")
    # There's no target used for this command, so we don't need to do anything
    # here for CMP0178.
    add_test(
      NAME ${test_name}
      COMMAND ${CMAKE_COMMAND}
      "-Darg_files_dest=${AST_FILES_DEST}"
      "-Darg_libs_dest=${AST_LIBS_DEST}"
      "-Darg_dev_test_dir=${AST_DEVICE_TEST_DIR}"
      "-Darg_dev_obj_store=${AST_DEVICE_OBJECT_STORE}"
      "-Darg_no_link_regex=${AST_NO_LINK_REGEX}"
      "-Darg_files=${processed_FILES}"
      "-Darg_libs=${AST_LIBS}"
      "-Darg_src_dir=${CMAKE_CURRENT_SOURCE_DIR}"
      -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/AndroidTestUtilities/PushToAndroidDevice.cmake)
  endif()
endfunction()
