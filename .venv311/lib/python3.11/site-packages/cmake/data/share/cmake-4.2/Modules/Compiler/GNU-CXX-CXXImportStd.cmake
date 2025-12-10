function (_cmake_cxx_import_std std variable)
  if (NOT CMAKE_CXX_STANDARD_LIBRARY STREQUAL "libstdc++")
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"Only `libstdc++` is supported\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  if (CMAKE_CXX_STDLIB_MODULES_JSON)
    set(_gnu_libstdcxx_modules_json_file "${CMAKE_CXX_STDLIB_MODULES_JSON}")
  else ()
    execute_process(
      COMMAND
        "${CMAKE_CXX_COMPILER}"
        ${CMAKE_CXX_COMPILER_ID_ARG1}
        -print-file-name=libstdc++.modules.json
      OUTPUT_VARIABLE _gnu_libstdcxx_modules_json_file
      ERROR_VARIABLE _gnu_libstdcxx_modules_json_file_err
      RESULT_VARIABLE _gnu_libstdcxx_modules_json_file_res
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE)
    if (_gnu_libstdcxx_modules_json_file_res)
      set("${variable}"
        "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"Could not find `libstdc++.modules.json` resource\")\n"
        PARENT_SCOPE)
      return ()
    endif ()
  endif ()

  # Without this file, we do not have modules installed.
  if (NOT EXISTS "${_gnu_libstdcxx_modules_json_file}")
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"`libstdc++.modules.json` resource does not exist\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  file(READ "${_gnu_libstdcxx_modules_json_file}" _gnu_libstdcxx_modules_json)
  string(JSON _gnu_modules_json_version GET "${_gnu_libstdcxx_modules_json}" "version")
  string(JSON _gnu_modules_json_revision GET "${_gnu_libstdcxx_modules_json}" "revision")
  # Require version 1.
  if (NOT _gnu_modules_json_version EQUAL "1")
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"`libstdc++.modules.json` version ${_gnu_modules_json_version}.${_gnu_modules_json_revision} is not recognized\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  string(JSON _gnu_modules_json_nmodules LENGTH "${_gnu_libstdcxx_modules_json}" "modules")
  # Don't declare the target without any modules.
  if (NOT _gnu_modules_json_nmodules)
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"`libstdc++.modules.json` does not list any available modules\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  # Declare the target.
  set(_gnu_libstdcxx_target "")
  string(APPEND _gnu_libstdcxx_target
    "add_library(__CMAKE::CXX${std} IMPORTED INTERFACE)\n")
  string(APPEND _gnu_libstdcxx_target
    "target_compile_features(__CMAKE::CXX${std} INTERFACE cxx_std_${std})\n")

  set(_gnu_modules_is_stdlib 0)
  set(_gnu_modules_include_dirs_list "")
  set(_gnu_modules_module_paths "")
  get_filename_component(_gnu_modules_dir "${_gnu_libstdcxx_modules_json_file}" DIRECTORY)

  # Add module sources.
  math(EXPR _gnu_modules_json_nmodules_range "${_gnu_modules_json_nmodules} - 1")
  foreach (_gnu_modules_json_modules_idx RANGE 0 "${_gnu_modules_json_nmodules_range}")
    string(JSON _gnu_modules_json_module GET "${_gnu_libstdcxx_modules_json}" "modules" "${_gnu_modules_json_modules_idx}")

    string(JSON _gnu_modules_json_module_source GET "${_gnu_modules_json_module}" "source-path")
    string(JSON _gnu_modules_json_module_is_stdlib GET "${_gnu_modules_json_module}" "is-std-library")
    string(JSON _gnu_modules_json_module_local_arguments ERROR_VARIABLE _gnu_modules_json_module_local_arguments_error GET "${_gnu_modules_json_module}" "local-arguments")
    string(JSON _gnu_modules_json_module_nsystem_include_directories ERROR_VARIABLE _gnu_modules_json_module_nsystem_include_directories_error LENGTH "${_gnu_modules_json_module_local_arguments}" "system-include-directories")

    if (_gnu_modules_json_module_local_arguments_error STREQUAL "NOTFOUND")
      set(_gnu_modules_json_module_local_arguments "")
    endif ()
    if (_gnu_modules_json_module_nsystem_include_directories_error STREQUAL "NOTFOUND")
      set(_gnu_modules_json_module_nsystem_include_directories 0)
    endif ()

    if (NOT IS_ABSOLUTE "${_gnu_modules_json_module_source}")
      string(PREPEND _gnu_modules_json_module_source "${_gnu_modules_dir}/")
    endif ()
    list(APPEND _gnu_modules_module_paths
      "${_gnu_modules_json_module_source}")

    if (_gnu_modules_json_module_is_stdlib)
      set(_gnu_modules_is_stdlib 1)
    endif ()

    if (_gnu_modules_json_module_nsystem_include_directories)
      math(EXPR _gnu_modules_json_module_nsystem_include_directories_range "${_gnu_modules_json_module_nsystem_include_directories} - 1")
      foreach (_gnu_modules_json_modules_system_include_directories_idx RANGE 0 "${_gnu_modules_json_module_nsystem_include_directories_range}")
        string(JSON _gnu_modules_json_module_system_include_directory GET "${_gnu_modules_json_module_local_arguments}" "system-include-directories" "${_gnu_modules_json_modules_system_include_directories_idx}")

        if (NOT IS_ABSOLUTE "${_gnu_modules_json_module_system_include_directory}")
          string(PREPEND _gnu_modules_json_module_system_include_directory "${_gnu_modules_dir}/")
        endif ()
        list(APPEND _gnu_modules_include_dirs_list
          "${_gnu_modules_json_module_system_include_directory}")
      endforeach ()
    endif ()
  endforeach ()

  # Split the paths into basedirs and module paths.
  set(_gnu_modules_base_dirs_list "")
  set(_gnu_modules_files "")
  foreach (_gnu_modules_module_path IN LISTS _gnu_modules_module_paths)
    get_filename_component(_gnu_module_dir "${_gnu_modules_module_path}" DIRECTORY)

    list(APPEND _gnu_modules_base_dirs_list
      "${_gnu_module_dir}")
    string(APPEND _gnu_modules_files
      " \"${_gnu_modules_module_path}\"")
  endforeach ()
  list(REMOVE_DUPLICATES _gnu_modules_base_dirs_list)
  set(_gnu_modules_base_dirs "")
  foreach (_gnu_modules_base_dir IN LISTS _gnu_modules_base_dirs_list)
    string(APPEND _gnu_modules_base_dirs
      " \"${_gnu_modules_base_dir}\"")
  endforeach ()

  # Create the file set for the modules.
  string(APPEND _gnu_libstdcxx_target
    "target_sources(__CMAKE::CXX${std}
  INTERFACE
  FILE_SET std TYPE CXX_MODULES
    BASE_DIRS ${_gnu_modules_base_dirs}
    FILES ${_gnu_modules_files})\n")

  set("${variable}" "${_gnu_libstdcxx_target}" PARENT_SCOPE)
endfunction ()
