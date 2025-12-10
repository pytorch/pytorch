# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

macro(cmake_nvcc_parse_implicit_info lang lang_var_)
  set(_nvcc_log "")
  string(REPLACE "\r" "" _nvcc_output_orig "${CMAKE_${lang}_COMPILER_PRODUCED_OUTPUT}")
  if(_nvcc_output_orig MATCHES "#\\\$ +PATH= *([^\n]*)\n")
    set(_nvcc_path "${CMAKE_MATCH_1}")
    string(APPEND _nvcc_log "  found 'PATH=' string: [${_nvcc_path}]\n")
    string(REPLACE ":" ";" _nvcc_path "${_nvcc_path}")
  else()
    set(_nvcc_path "")
    string(REPLACE "\n" "\n    " _nvcc_output_log "\n${_nvcc_output_orig}")
    string(APPEND _nvcc_log "  no 'PATH=' string found in nvcc output:${_nvcc_output_log}\n")
  endif()
  if(_nvcc_output_orig MATCHES "#\\\$ +LIBRARIES= *([^\n]*)\n")
    set(_nvcc_libraries "${CMAKE_MATCH_1}")
    string(APPEND _nvcc_log "  found 'LIBRARIES=' string: [${_nvcc_libraries}]\n")
  else()
    set(_nvcc_libraries "")
    string(REPLACE "\n" "\n    " _nvcc_output_log "\n${_nvcc_output_orig}")
    string(APPEND _nvcc_log "  no 'LIBRARIES=' string found in nvcc output:${_nvcc_output_log}\n")
  endif()
  if(_nvcc_output_orig MATCHES "#\\\$ +INCLUDES= *([^\n]*)\n")
    set(_nvcc_includes "${CMAKE_MATCH_1}")
    string(APPEND _nvcc_log "  found 'INCLUDES=' string: [${_nvcc_includes}]\n")
  else()
    set(_nvcc_includes "")
    string(REPLACE "\n" "\n    " _nvcc_output_log "\n${_nvcc_output_orig}")
    string(APPEND _nvcc_log "  no 'INCLUDES=' string found in nvcc output:${_nvcc_output_log}\n")
  endif()
  if(_nvcc_output_orig MATCHES "#\\\$ +SYSTEM_INCLUDES= *([^\n]*)\n")
    set(_nvcc_system_includes "${CMAKE_MATCH_1}")
    string(APPEND _nvcc_log "  found 'SYSTEM_INCLUDES=' string: [${_nvcc_system_includes}]\n")
  else()
    set(_nvcc_system_includes "")
    string(REPLACE "\n" "\n    " _nvcc_output_log "\n${_nvcc_output_orig}")
    string(APPEND _nvcc_log "  no 'SYSTEM_INCLUDES=' string found in nvcc output:${_nvcc_output_log}\n")
  endif()
  string(REGEX MATCHALL "-arch compute_([0-9]+)" _nvcc_target_cpus "${_nvcc_output_orig}")
  foreach(_nvcc_target_cpu ${_nvcc_target_cpus})
    if(_nvcc_target_cpu MATCHES "-arch compute_([0-9]+)")
      list(APPEND CMAKE_${lang}_ARCHITECTURES_DEFAULT "${CMAKE_MATCH_1}")
    endif()
  endforeach()

  set(_nvcc_link_line "")
  if(_nvcc_libraries)
    # Remove variable assignments.
    string(REGEX REPLACE "#\\\$ *[^= ]+=[^\n]*\n" "" _nvcc_output "${_nvcc_output_orig}")
    # Encode [] characters that break list expansion.
    string(REPLACE "[" "{==={" _nvcc_output "${_nvcc_output}")
    string(REPLACE "]" "}===}" _nvcc_output "${_nvcc_output}")
    # Split lines.
    string(REGEX REPLACE "\n+(#\\\$ )?" ";" _nvcc_output "${_nvcc_output}")
    foreach(line IN LISTS _nvcc_output)
      set(_nvcc_output_line "${line}")
      string(REPLACE "{==={" "[" _nvcc_output_line "${_nvcc_output_line}")
      string(REPLACE "}===}" "]" _nvcc_output_line "${_nvcc_output_line}")
      string(APPEND _nvcc_log "  considering line: [${_nvcc_output_line}]\n")
      if("${_nvcc_output_line}" MATCHES "^ *nvlink")
        string(APPEND _nvcc_log "    ignoring nvlink line\n")
      elseif("${_nvcc_output_line}" MATCHES "(link\\.exe .*CompilerId${lang}\\.exe.*)$")
        set(_nvcc_link_line "${CMAKE_MATCH_1}")
        string(APPEND _nvcc_log "    extracted link line: [${_nvcc_link_line}]\n")
      elseif(_nvcc_libraries)
        if("${_nvcc_output_line}" MATCHES "(@\"?((tmp/)?a\\.exe\\.res)\"?)")
          set(_nvcc_link_res_arg "${CMAKE_MATCH_1}")
          set(_nvcc_link_res_file "${CMAKE_MATCH_2}")
          set(_nvcc_link_res "${CMAKE_PLATFORM_INFO_DIR}/CompilerId${lang}/${_nvcc_link_res_file}")
          if(EXISTS "${_nvcc_link_res}")
            file(READ "${_nvcc_link_res}" _nvcc_link_res_content)
            string(REPLACE "${_nvcc_link_res_arg}" "${_nvcc_link_res_content}" _nvcc_output_line "${_nvcc_output_line}")
          endif()
        endif()
        string(FIND "${_nvcc_output_line}" "${_nvcc_libraries}" _nvcc_libraries_pos)
        if(NOT _nvcc_libraries_pos EQUAL -1)
          set(_nvcc_link_line "${_nvcc_output_line}")
          string(APPEND _nvcc_log "    extracted link line: [${_nvcc_link_line}]\n")
        endif()
      endif()
    endforeach()
  endif()

  if(_nvcc_link_line)
    if("x${CMAKE_${lang}_SIMULATE_ID}" STREQUAL "xMSVC")
      set(CMAKE_${lang}_HOST_LINK_LAUNCHER "${CMAKE_LINKER}")
    else()
      #extract the compiler that is being used for linking
      separate_arguments(_nvcc_link_line_args UNIX_COMMAND "${_nvcc_link_line}")
      list(GET _nvcc_link_line_args 0 _nvcc_host_link_launcher)
      if(IS_ABSOLUTE "${_nvcc_host_link_launcher}")
        string(APPEND _nvcc_log "  extracted link launcher absolute path: [${_nvcc_host_link_launcher}]\n")
        set(CMAKE_${lang}_HOST_LINK_LAUNCHER "${_nvcc_host_link_launcher}")
      else()
        string(APPEND _nvcc_log "  extracted link launcher name: [${_nvcc_host_link_launcher}]\n")
        find_program(_nvcc_find_host_link_launcher
          NAMES ${_nvcc_host_link_launcher}
          PATHS ${_nvcc_path} NO_DEFAULT_PATH)
        find_program(_nvcc_find_host_link_launcher
          NAMES ${_nvcc_host_link_launcher})
        if(_nvcc_find_host_link_launcher)
          string(APPEND _nvcc_log "  found link launcher absolute path: [${_nvcc_find_host_link_launcher}]\n")
          set(CMAKE_${lang}_HOST_LINK_LAUNCHER "${_nvcc_find_host_link_launcher}")
        else()
          string(APPEND _nvcc_log "  could not find link launcher absolute path\n")
          set(CMAKE_${lang}_HOST_LINK_LAUNCHER "${_nvcc_host_link_launcher}")
        endif()
        unset(_nvcc_find_host_link_launcher CACHE)
      endif()
    endif()

    #prefix the line with cuda-fake-ld so that implicit link info believes it is
    #a link line
    set(_nvcc_link_line "cuda-fake-ld ${_nvcc_link_line}")
    CMAKE_PARSE_IMPLICIT_LINK_INFO("${_nvcc_link_line}"
                                   CMAKE_${lang}_HOST_IMPLICIT_LINK_LIBRARIES
                                   CMAKE_${lang}_HOST_IMPLICIT_LINK_DIRECTORIES
                                   CMAKE_${lang}_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES
                                   log
                                   "${CMAKE_${lang}_IMPLICIT_OBJECT_REGEX}"
                                   LANGUAGE ${lang})

    # Detect CMAKE_${lang}_RUNTIME_LIBRARY_DEFAULT from the compiler by looking at which
    # cudart library exists in the implicit link libraries passed to the host linker.
    # This is required when a project sets the cuda runtime library as part of the
    # initial flags.
    if(";${CMAKE_${lang}_HOST_IMPLICIT_LINK_LIBRARIES};" MATCHES [[;cudart_static(\.lib)?;]])
      set(CMAKE_${lang}_RUNTIME_LIBRARY_DEFAULT "STATIC")
    elseif(";${CMAKE_${lang}_HOST_IMPLICIT_LINK_LIBRARIES};" MATCHES [[;cudart(\.lib)?;]])
      set(CMAKE_${lang}_RUNTIME_LIBRARY_DEFAULT "SHARED")
    else()
      set(CMAKE_${lang}_RUNTIME_LIBRARY_DEFAULT "NONE")
    endif()

    message(CONFIGURE_LOG
      "Parsed ${lang} nvcc implicit link information:\n${_nvcc_log}\n${log}\n\n")
  else()
    message(CONFIGURE_LOG
      "Failed to parse ${lang} nvcc implicit link information:\n${_nvcc_log}\n\n")
    message(FATAL_ERROR "Failed to extract nvcc implicit link line.")
  endif()

  set(${lang_var_}TOOLKIT_INCLUDE_DIRECTORIES)
  if(_nvcc_includes OR _nvcc_system_includes)
    # across all operating system each include directory is prefixed with -I
    separate_arguments(_nvcc_output NATIVE_COMMAND "${_nvcc_includes}")
    foreach(line IN LISTS _nvcc_output)
      string(REGEX REPLACE "^-I" "" line "${line}")
      get_filename_component(line "${line}" ABSOLUTE)
      list(APPEND ${lang_var_}TOOLKIT_INCLUDE_DIRECTORIES "${line}")
    endforeach()

    # across all operating system each system include directory is prefixed with -isystem
    unset(_nvcc_output)
    separate_arguments(_nvcc_output NATIVE_COMMAND "${_nvcc_system_includes}")
    foreach(line IN LISTS _nvcc_output)
      string(REGEX REPLACE "^-isystem" "" line "${line}")
      if(line)
        get_filename_component(line "${line}" ABSOLUTE)
        list(APPEND ${lang_var_}TOOLKIT_INCLUDE_DIRECTORIES "${line}")
      endif()
    endforeach()

    message(CONFIGURE_LOG
      "Parsed CUDA nvcc include information:\n${_nvcc_log}\n${log}\n\n")
  else()
    message(CONFIGURE_LOG
      "Failed to detect CUDA nvcc include information:\n${_nvcc_log}\n\n")
  endif()
endmacro()
