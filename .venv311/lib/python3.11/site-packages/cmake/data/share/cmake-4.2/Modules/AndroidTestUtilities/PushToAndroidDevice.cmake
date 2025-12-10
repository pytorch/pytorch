# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This function handles pushing all of the test files needed to the device.
# It places the data files in the object store and makes links to them from
# the appropriate directories.
#
# This function accepts the following named parameters:
# DIRS          : one or more directories needed for testing.
# FILES         : one or more files needed for testing.
# LIBS          : one or more libraries needed for testing.
# DIRS_DEST     : specify where the directories should be installed.
# FILES_DEST    : specify where the files should be installed.
# LIBS_DEST     : specify where the libraries should be installed.
# DEV_OBJ_STORE : specify where the actual data files should be placed.
# DEV_TEST_DIR  : specify the root file for the module test directory.
# The DEV_OBJ_STORE and DEV_TEST_DIR variables are required.

# The parameters to this function should be set to the list of directories,
# files, and libraries that need to be installed prior to testing.
function(android_push_test_files_to_device)

  # The functions in the module need the adb executable.
  find_program(adb_executable adb)
  if(NOT adb_executable)
    message(FATAL_ERROR "could not find adb")
  endif()

  function(execute_adb_command)
    execute_process(COMMAND ${adb_executable} ${ARGN} RESULT_VARIABLE res_var OUTPUT_VARIABLE out_var ERROR_VARIABLE err_var)
    set(out_var ${out_var} PARENT_SCOPE)
    if(res_var)
      string(REGEX REPLACE ";" " " com "${ARGN}")
      message(FATAL_ERROR "Error occurred during adb command: adb ${com}\nError: ${err_var}.")
    endif()
  endfunction()

  # Checks to make sure that a given file exists on the device. If it does,
  # if(file_exists) will return true.
  macro(check_device_file_exists device_file file_exists)
    set(${file_exists} "")
    execute_process(
      COMMAND ${adb_executable} shell ls ${device_file}
      OUTPUT_VARIABLE out_var ERROR_VARIABLE out_var)
    if(NOT out_var) # when a directory exists but is empty the output is empty
      set(${file_exists} "YES")
    else()
      string(FIND ${out_var} "No such file or directory" no_file_exists)
      if(${no_file_exists} STREQUAL "-1") # -1 means the file exists
        set(${file_exists} "YES")
      endif()
    endif()
  endmacro()

  # Checks to see if a filename matches a regex.
  function(filename_regex filename reg_ex)
    string(REGEX MATCH ${reg_ex} filename_match ${filename})
    set(filename_match ${filename_match} PARENT_SCOPE)
  endfunction()

  # If a file with given name exists in the CMAKE_BINARY_DIR then use that file.
  # Otherwise use the file with root in CMAKE_CURRENT_SOURCE_DIR.
  macro(set_absolute_path relative_path absolute_path)
    set(${absolute_path} ${arg_src_dir}/${relative_path})
    if(EXISTS ${CMAKE_BINARY_DIR}/${relative_path})
      set(${absolute_path} ${CMAKE_BINARY_DIR}/${relative_path})
    endif()
    if(NOT EXISTS ${${absolute_path}})
      if(EXISTS ${relative_path})
        set(${absolute_path} ${relative_path})
      else()
        message(FATAL_ERROR "Cannot find file for specified path: ${relative_path}")
      endif()
    endif()
  endmacro()

  # This function pushes the data into the device object store and
  # creates a link to that data file in a specified location.
  #
  # This function requires the following un-named parameters:
  # data_path        : absolute path to data to load into dev obj store.
  # dev_object_store : absolute path to the device object store directory.
  # link_origin      : absolute path to the origin of the link to the dev obj store data file.
  function(push_and_link data_path dev_object_store link_origin)
    file(SHA1 ${data_path} hash_val)
    set(obj_store_dst ${dev_object_store}/${hash_val})
    check_device_file_exists(${obj_store_dst} obj_store_file_exists)
    # TODO: Verify that the object store file is indeed hashed correctly. Could use md5.
    if(NOT obj_store_file_exists)
      execute_adb_command(push ${data_path} ${obj_store_dst})
    endif()
    check_device_file_exists(${link_origin} link_exists)
    if(link_exists)
      execute_adb_command(shell rm -f ${link_origin})
    endif()
    foreach(ex ${arg_no_link_regex})
      filename_regex(${data_path} ${ex})
      list(APPEND match_ex ${filename_match})
    endforeach()
    if(match_ex)
      execute_adb_command(shell cp ${obj_store_dst} ${link_origin})
    else()
      execute_adb_command(shell ln -s ${obj_store_dst} ${link_origin})
    endif()
  endfunction()

  #----------------------------------------------------------------------------
  #--------------------Beginning of actual function----------------------------
  #----------------------------------------------------------------------------
  set(oneValueArgs FILES_DEST LIBS_DEST DEV_TEST_DIR DEV_OBJ_STORE)
  set(multiValueArgs FILES LIBS)
  cmake_parse_arguments(_arg "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Setup of object store and test dir.
  check_device_file_exists(${_arg_DEV_OBJ_STORE} dev_obj_store_exists)
  if(NOT dev_obj_store_exists)
    execute_adb_command(shell mkdir -p ${_arg_DEV_OBJ_STORE})
  endif()
  check_device_file_exists(${_arg_DEV_TEST_DIR} test_dir_exists)
  if(test_dir_exists)
    # This is protected in the SetupProjectTests module.
    execute_adb_command(shell rm -r ${_arg_DEV_TEST_DIR})
  endif()
  execute_adb_command(shell mkdir -p ${_arg_DEV_TEST_DIR})

  # Looping over the various types of test data possible.
  foreach(TYPE ${multiValueArgs})
    if(_arg_${TYPE})

      # determine if the data type destination has been explicitly specified.
      if(_arg_${TYPE}_DEST)
        set(dest ${_arg_${TYPE}_DEST})
      else()
        if(${TYPE} STREQUAL LIBS)
          set(dest ${_arg_DEV_TEST_DIR}/lib)
        else()
          set(dest ${_arg_DEV_TEST_DIR})
        endif()
      endif()
      execute_adb_command(shell mkdir -p ${dest})

      # Loop over the files passed in
      foreach(relative_path ${_arg_${TYPE}})
        # The absolute path can be through the source directory or the build directory.
        # If the file/dir exists in the build directory that version is chosen.
        set_absolute_path(${relative_path} absolute_path)
        # Need to transfer all data files in the data directories to the device
        # except those explicitly ignored.
        if(${TYPE} STREQUAL FILES)
          get_filename_component(file_dir ${relative_path} DIRECTORY)
          # dest was determined earlier, relative_path is a dir, file is path from relative path to a data
          set(cur_dest ${dest}/${relative_path})
          set(on_dev_dir ${dest}/${file_dir})
          execute_adb_command(shell mkdir -p ${on_dev_dir})
          if(IS_SYMLINK ${absolute_path})
            get_filename_component(real_data_origin ${absolute_path} REALPATH)
            push_and_link(${real_data_origin} ${_arg_DEV_OBJ_STORE} ${cur_dest})
          else()
            push_and_link(${absolute_path} ${_arg_DEV_OBJ_STORE} ${cur_dest})
          endif()
        else() # LIBS
          execute_adb_command(push ${absolute_path} ${dest})
        endif()
      endforeach()
    endif()
  endforeach()
endfunction()

android_push_test_files_to_device(
  FILES_DEST ${arg_files_dest}
  LIBS_DEST ${arg_libs_dest}
  DEV_TEST_DIR ${arg_dev_test_dir}
  DEV_OBJ_STORE ${arg_dev_obj_store}
  FILES ${arg_files}
  LIBS ${arg_libs}
  )
