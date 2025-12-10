function (_cmake_cxx_import_std std variable)
  if (CMAKE_CXX_STANDARD_LIBRARY STREQUAL "libc++")
    set(_clang_modules_json_impl "libc++")
  elseif (CMAKE_CXX_STANDARD_LIBRARY STREQUAL "libstdc++")
    set(_clang_modules_json_impl "libstdc++")
  else ()
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"Only `libc++` and `libstdc++` are supported\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  if (CMAKE_CXX_STDLIB_MODULES_JSON)
    set(_clang_libcxx_modules_json_file "${CMAKE_CXX_STDLIB_MODULES_JSON}")
  else ()
    execute_process(
      COMMAND
        "${CMAKE_CXX_COMPILER}"
        ${CMAKE_CXX_COMPILER_ID_ARG1}
        "-print-file-name=${_clang_modules_json_impl}.modules.json"
      OUTPUT_VARIABLE _clang_libcxx_modules_json_file
      ERROR_VARIABLE _clang_libcxx_modules_json_file_err
      RESULT_VARIABLE _clang_libcxx_modules_json_file_res
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE)
    if (_clang_libcxx_modules_json_file_res)
      set("${variable}"
        "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"Could not find `${_clang_modules_json_impl}.modules.json` resource\")\n"
        PARENT_SCOPE)
      return ()
    endif ()
  endif ()

  # Without this file, we do not have modules installed.
  if (NOT EXISTS "${_clang_libcxx_modules_json_file}")
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"`${_clang_modules_json_impl}.modules.json` resource does not exist\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "18.1.2" AND
      CMAKE_CXX_STANDARD_LIBRARY STREQUAL "libc++")
    # The original PR had a key spelling mismatch internally. Do not support it
    # and instead require a release known to have the fix.
    # https://github.com/llvm/llvm-project/pull/83036
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"LLVM 18.1.2 is required for `${_clang_modules_json_impl}.modules.json` format fix\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  file(READ "${_clang_libcxx_modules_json_file}" _clang_libcxx_modules_json)
  string(JSON _clang_modules_json_version GET "${_clang_libcxx_modules_json}" "version")
  string(JSON _clang_modules_json_revision GET "${_clang_libcxx_modules_json}" "revision")
  # Require version 1.
  if (NOT _clang_modules_json_version EQUAL "1")
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"`libc++.modules.json` version ${_clang_modules_json_version}.${_clang_modules_json_revision} is not recognized\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  string(JSON _clang_modules_json_nmodules LENGTH "${_clang_libcxx_modules_json}" "modules")
  # Don't declare the target without any modules.
  if (NOT _clang_modules_json_nmodules)
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"`libc++.modules.json` does not list any available modules\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  # Declare the target.
  set(_clang_libcxx_target "")
  # Clang 18 does not provide the module initializer for the `std` modules.
  # Create a static library to hold these. Hope that Clang 19 can provide this,
  # but never run the code.
  string(APPEND _clang_libcxx_target
    "add_library(__cmake_cxx${std} STATIC)\n")
  string(APPEND _clang_libcxx_target
    "target_sources(__cmake_cxx${std} INTERFACE \"$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,STATIC_LIBRARY>:$<TARGET_OBJECTS:__cmake_cxx${std}>>\")\n")
  string(APPEND _clang_libcxx_target
    "set_property(TARGET __cmake_cxx${std} PROPERTY EXCLUDE_FROM_ALL 1)\n")
  string(APPEND _clang_libcxx_target
    "set_property(TARGET __cmake_cxx${std} PROPERTY CXX_SCAN_FOR_MODULES 1)\n")
  string(APPEND _clang_libcxx_target
    "set_property(TARGET __cmake_cxx${std} PROPERTY CXX_MODULE_STD 0)\n")
  string(APPEND _clang_libcxx_target
    "target_compile_features(__cmake_cxx${std} PUBLIC cxx_std_${std})\n")

  set(_clang_modules_is_stdlib 0)
  set(_clang_modules_include_dirs_list "")
  set(_clang_modules_module_paths "")
  get_filename_component(_clang_modules_dir "${_clang_libcxx_modules_json_file}" DIRECTORY)

  # Add module sources.
  math(EXPR _clang_modules_json_nmodules_range "${_clang_modules_json_nmodules} - 1")
  foreach (_clang_modules_json_modules_idx RANGE 0 "${_clang_modules_json_nmodules_range}")
    string(JSON _clang_modules_json_module GET "${_clang_libcxx_modules_json}" "modules" "${_clang_modules_json_modules_idx}")

    string(JSON _clang_modules_json_module_source GET "${_clang_modules_json_module}" "source-path")
    string(JSON _clang_modules_json_module_is_stdlib GET "${_clang_modules_json_module}" "is-std-library")
    string(JSON _clang_modules_json_module_local_arguments ERROR_VARIABLE _clang_modules_json_module_local_arguments_error GET "${_clang_modules_json_module}" "local-arguments")
    string(JSON _clang_modules_json_module_nsystem_include_directories ERROR_VARIABLE _clang_modules_json_module_nsystem_include_directories_error LENGTH "${_clang_modules_json_module_local_arguments}" "system-include-directories")

    if (_clang_modules_json_module_local_arguments_error)
      set(_clang_modules_json_module_local_arguments "")
    endif ()
    if (_clang_modules_json_module_nsystem_include_directories_error)
      set(_clang_modules_json_module_nsystem_include_directories 0)
    endif ()
    if (NOT IS_ABSOLUTE "${_clang_modules_json_module_source}")
      string(PREPEND _clang_modules_json_module_source "${_clang_modules_dir}/")
    endif ()
    list(APPEND _clang_modules_module_paths
      "${_clang_modules_json_module_source}")

    if (_clang_modules_json_module_is_stdlib)
      set(_clang_modules_is_stdlib 1)
    endif ()

    if (_clang_modules_json_module_nsystem_include_directories)
      math(EXPR _clang_modules_json_module_nsystem_include_directories_range "${_clang_modules_json_module_nsystem_include_directories} - 1")
      foreach (_clang_modules_json_modules_system_include_directories_idx RANGE 0 "${_clang_modules_json_module_nsystem_include_directories_range}")
        string(JSON _clang_modules_json_module_system_include_directory GET "${_clang_modules_json_module_local_arguments}" "system-include-directories" "${_clang_modules_json_modules_system_include_directories_idx}")

        if (NOT IS_ABSOLUTE "${_clang_modules_json_module_system_include_directory}")
          string(PREPEND _clang_modules_json_module_system_include_directory "${_clang_modules_dir}/")
        endif ()
        list(APPEND _clang_modules_include_dirs_list
          "${_clang_modules_json_module_system_include_directory}")
      endforeach ()
    endif ()
  endforeach ()

  # Split the paths into basedirs and module paths.
  set(_clang_modules_base_dirs_list "")
  set(_clang_modules_files "")
  foreach (_clang_modules_module_path IN LISTS _clang_modules_module_paths)
    get_filename_component(_clang_module_dir "${_clang_modules_module_path}" DIRECTORY)

    list(APPEND _clang_modules_base_dirs_list
      "${_clang_module_dir}")
    string(APPEND _clang_modules_files
      " \"${_clang_modules_module_path}\"")
  endforeach ()
  list(REMOVE_DUPLICATES _clang_modules_base_dirs_list)
  set(_clang_modules_base_dirs "")
  foreach (_clang_modules_base_dir IN LISTS _clang_modules_base_dirs_list)
    string(APPEND _clang_modules_base_dirs
      " \"${_clang_modules_base_dir}\"")
  endforeach ()

  # If we have a standard library module, suppress warnings about reserved
  # module names.
  if (_clang_modules_is_stdlib)
    string(APPEND _clang_libcxx_target
      "target_compile_options(__cmake_cxx${std} PRIVATE -Wno-reserved-module-identifier)\n")
  endif ()

  # Set up include directories.
  list(REMOVE_DUPLICATES _clang_modules_include_dirs_list)
  set(_clang_modules_include_dirs "")
  foreach (_clang_modules_include_dir IN LISTS _clang_modules_include_dirs_list)
    string(APPEND _clang_modules_include_dirs
      " \"${_clang_modules_include_dir}\"")
  endforeach ()
  string(APPEND _clang_libcxx_target
    "target_include_directories(__cmake_cxx${std} PRIVATE ${_clang_modules_include_dirs})\n")

  # Create the file set for the modules.
  string(APPEND _clang_libcxx_target
    "target_sources(__cmake_cxx${std}
  PUBLIC
  FILE_SET std TYPE CXX_MODULES
    BASE_DIRS ${_clang_modules_base_dirs}
    FILES ${_clang_modules_files})\n")

  # Wrap the `__cmake_cxx${std}` target in a check.
  string(PREPEND _clang_libcxx_target
    "if (NOT TARGET \"__cmake_cxx${std}\")\n")
  string(APPEND _clang_libcxx_target
    "endif ()\n")
  string(APPEND _clang_libcxx_target
    "add_library(__CMAKE::CXX${std} ALIAS __cmake_cxx${std})\n")

  set("${variable}" "${_clang_libcxx_target}" PARENT_SCOPE)
endfunction ()
