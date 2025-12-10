function (_cmake_cxx_import_std std variable)
  if (CMAKE_CXX_STDLIB_MODULES_JSON)
    set(_msvc_modules_json_file "${CMAKE_CXX_STDLIB_MODULES_JSON}")
  else ()
    find_file(_msvc_modules_json_file
      NAME modules.json
      HINTS
        "$ENV{VCToolsInstallDir}/modules"
      PATHS
        "$ENV{INCLUDE}"
        "${CMAKE_CXX_COMPILER}/../../.."
        "${CMAKE_CXX_COMPILER}/../.."    # msvc-wine layout
      PATH_SUFFIXES
        ../modules
      NO_CACHE)
    # Without this file, we do not have modules installed.
    if (NOT EXISTS "${_msvc_modules_json_file}")
      set("${variable}"
        "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"Could not find `modules.json` resource\")\n"
        PARENT_SCOPE)
      return ()
    endif ()
  endif ()

  file(READ "${_msvc_modules_json_file}" _msvc_modules_json)
  string(JSON _msvc_json_version GET "${_msvc_modules_json}" "version")
  string(JSON _msvc_json_revision GET "${_msvc_modules_json}" "revision")
  # Require version 1.
  if (NOT _msvc_json_version EQUAL "1")
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"`modules.json` version ${_msvc_json_version}.${_msvc_json_revision} is not recognized\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  string(JSON _msvc_json_library GET "${_msvc_modules_json}" "library")
  # Bail if we don't understand the library.
  if (NOT _msvc_json_library STREQUAL "microsoft/STL")
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"`modules.json` library `${_msvc_json_library}` is not recognized\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  string(JSON _msvc_json_nmodules LENGTH "${_msvc_modules_json}" "module-sources")
  # Don't declare the target without any modules.
  if (NOT _msvc_json_nmodules)
    set("${variable}"
      "set(CMAKE_CXX${std}_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE \"`modules.json` does not list any available modules\")\n"
      PARENT_SCOPE)
    return ()
  endif ()

  # Declare the target.
  set(_msvc_std_target "")
  string(APPEND _msvc_std_target
    "add_library(__cmake_cxx${std} STATIC)\n")
  string(APPEND _msvc_std_target
    "target_sources(__cmake_cxx${std} INTERFACE \"$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,STATIC_LIBRARY>:$<TARGET_OBJECTS:__cmake_cxx${std}>>\")\n")
  string(APPEND _msvc_std_target
    "set_property(TARGET __cmake_cxx${std} PROPERTY EXCLUDE_FROM_ALL 1)\n")
  string(APPEND _msvc_std_target
    "set_property(TARGET __cmake_cxx${std} PROPERTY CXX_SCAN_FOR_MODULES 1)\n")
  string(APPEND _msvc_std_target
    "set_property(TARGET __cmake_cxx${std} PROPERTY CXX_MODULE_STD 0)\n")
  string(APPEND _msvc_std_target
    "target_compile_features(__cmake_cxx${std} PUBLIC cxx_std_${std})\n")

  set(_msvc_modules_module_paths "")
  get_filename_component(_msvc_modules_dir "${_msvc_modules_json_file}" DIRECTORY)

  # Add module sources.
  math(EXPR _msvc_modules_json_nmodules_range "${_msvc_json_nmodules} - 1")
  foreach (_msvc_modules_json_modules_idx RANGE 0 "${_msvc_modules_json_nmodules_range}")
    string(JSON _msvc_modules_json_module_source GET "${_msvc_modules_json}" "module-sources" "${_msvc_modules_json_modules_idx}")

    if (NOT IS_ABSOLUTE "${_msvc_modules_json_module_source}")
      string(PREPEND _msvc_modules_json_module_source "${_msvc_modules_dir}/")
    endif ()
    list(APPEND _msvc_modules_module_paths
      "${_msvc_modules_json_module_source}")
  endforeach ()

  # Split the paths into basedirs and module paths.
  set(_msvc_modules_base_dirs_list "")
  set(_msvc_modules_files "")
  foreach (_msvc_modules_module_path IN LISTS _msvc_modules_module_paths)
    get_filename_component(_msvc_module_dir "${_msvc_modules_module_path}" DIRECTORY)

    list(APPEND _msvc_modules_base_dirs_list
      "${_msvc_module_dir}")
    string(APPEND _msvc_modules_files
      " \"${_msvc_modules_module_path}\"")
  endforeach ()
  list(REMOVE_DUPLICATES _msvc_modules_base_dirs_list)
  set(_msvc_modules_base_dirs "")
  foreach (_msvc_modules_base_dir IN LISTS _msvc_modules_base_dirs_list)
    string(APPEND _msvc_modules_base_dirs
      " \"${_msvc_modules_base_dir}\"")
  endforeach ()

  # Create the file set for the modules.
  string(APPEND _msvc_std_target
    "target_sources(__cmake_cxx${std}
  PUBLIC
  FILE_SET std TYPE CXX_MODULES
    BASE_DIRS ${_msvc_modules_base_dirs}
    FILES ${_msvc_modules_files})\n")

  # Wrap the `__cmake_cxx${std}` target in a check.
  string(PREPEND _msvc_std_target
    "if (NOT TARGET \"__cmake_cxx${std}\")\n")
  string(APPEND _msvc_std_target
    "endif ()\n")
  string(APPEND _msvc_std_target
    "add_library(__CMAKE::CXX${std} ALIAS __cmake_cxx${std})\n")

  set("${variable}" "${_msvc_std_target}" PARENT_SCOPE)
endfunction ()
