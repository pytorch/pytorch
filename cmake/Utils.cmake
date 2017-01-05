################################################################################################
# Command alias for debugging messages
# Usage:
#   dmsg(<message>)
function(dmsg)
  message(STATUS ${ARGN})
endfunction()

################################################################################################
# Removes duplicates from list(s)
# Usage:
#   caffe_list_unique(<list_variable> [<list_variable>] [...])
macro(caffe_list_unique)
  foreach(__lst ${ARGN})
    if(${__lst})
      list(REMOVE_DUPLICATES ${__lst})
    endif()
  endforeach()
endmacro()

################################################################################################
# Clears variables from list
# Usage:
#   caffe_clear_vars(<variables_list>)
macro(caffe_clear_vars)
  foreach(_var ${ARGN})
    unset(${_var})
  endforeach()
endmacro()

################################################################################################
# Removes duplicates from string
# Usage:
#   caffe_string_unique(<string_variable>)
function(caffe_string_unique __string)
  if(${__string})
    set(__list ${${__string}})
    separate_arguments(__list)
    list(REMOVE_DUPLICATES __list)
    foreach(__e ${__list})
      set(__str "${__str} ${__e}")
    endforeach()
    set(${__string} ${__str} PARENT_SCOPE)
  endif()
endfunction()

################################################################################################
# Prints list element per line
# Usage:
#   caffe_print_list(<list>)
function(caffe_print_list)
  foreach(e ${ARGN})
    message(STATUS ${e})
  endforeach()
endfunction()

################################################################################################
# Function merging lists of compiler flags to single string.
# Usage:
#   caffe_merge_flag_lists(out_variable <list1> [<list2>] [<list3>] ...)
function(caffe_merge_flag_lists out_var)
  set(__result "")
  foreach(__list ${ARGN})
    foreach(__flag ${${__list}})
      string(STRIP ${__flag} __flag)
      set(__result "${__result} ${__flag}")
    endforeach()
  endforeach()
  string(STRIP ${__result} __result)
  set(${out_var} ${__result} PARENT_SCOPE)
endfunction()

################################################################################################
# Converts all paths in list to absolute
# Usage:
#   caffe_convert_absolute_paths(<list_variable>)
function(caffe_convert_absolute_paths variable)
  set(__dlist "")
  foreach(__s ${${variable}})
    get_filename_component(__abspath ${__s} ABSOLUTE)
    list(APPEND __list ${__abspath})
  endforeach()
  set(${variable} ${__list} PARENT_SCOPE)
endfunction()

################################################################################################
# Reads set of version defines from the header file
# Usage:
#   caffe_parse_header(<file> <define1> <define2> <define3> ..)
macro(caffe_parse_header FILENAME FILE_VAR)
  set(vars_regex "")
  set(__parnet_scope OFF)
  set(__add_cache OFF)
  foreach(name ${ARGN})
    if("${name}" STREQUAL "PARENT_SCOPE")
      set(__parnet_scope ON)
    elseif("${name}" STREQUAL "CACHE")
      set(__add_cache ON)
    elseif(vars_regex)
      set(vars_regex "${vars_regex}|${name}")
    else()
      set(vars_regex "${name}")
    endif()
  endforeach()
  if(EXISTS "${FILENAME}")
    file(STRINGS "${FILENAME}" ${FILE_VAR} REGEX "#define[ \t]+(${vars_regex})[ \t]+[0-9]+" )
  else()
    unset(${FILE_VAR})
  endif()
  foreach(name ${ARGN})
    if(NOT "${name}" STREQUAL "PARENT_SCOPE" AND NOT "${name}" STREQUAL "CACHE")
      if(${FILE_VAR})
        if(${FILE_VAR} MATCHES ".+[ \t]${name}[ \t]+([0-9]+).*")
          string(REGEX REPLACE ".+[ \t]${name}[ \t]+([0-9]+).*" "\\1" ${name} "${${FILE_VAR}}")
        else()
          set(${name} "")
        endif()
        if(__add_cache)
          set(${name} ${${name}} CACHE INTERNAL "${name} parsed from ${FILENAME}" FORCE)
        elseif(__parnet_scope)
          set(${name} "${${name}}" PARENT_SCOPE)
        endif()
      else()
        unset(${name} CACHE)
      endif()
    endif()
  endforeach()
endmacro()

################################################################################################
# Reads single version define from the header file and parses it
# Usage:
#   caffe_parse_header_single_define(<library_name> <file> <define_name>)
function(caffe_parse_header_single_define LIBNAME HDR_PATH VARNAME)
  set(${LIBNAME}_H "")
  if(EXISTS "${HDR_PATH}")
    file(STRINGS "${HDR_PATH}" ${LIBNAME}_H REGEX "^#define[ \t]+${VARNAME}[ \t]+\"[^\"]*\".*$" LIMIT_COUNT 1)
  endif()

  if(${LIBNAME}_H)
    string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MAJOR "${${LIBNAME}_H}")
    string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MINOR  "${${LIBNAME}_H}")
    string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_PATCH "${${LIBNAME}_H}")
    set(${LIBNAME}_VERSION_MAJOR ${${LIBNAME}_VERSION_MAJOR} ${ARGN} PARENT_SCOPE)
    set(${LIBNAME}_VERSION_MINOR ${${LIBNAME}_VERSION_MINOR} ${ARGN} PARENT_SCOPE)
    set(${LIBNAME}_VERSION_PATCH ${${LIBNAME}_VERSION_PATCH} ${ARGN} PARENT_SCOPE)
    set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_MAJOR}.${${LIBNAME}_VERSION_MINOR}.${${LIBNAME}_VERSION_PATCH}" PARENT_SCOPE)

    # append a TWEAK version if it exists:
    set(${LIBNAME}_VERSION_TWEAK "")
    if("${${LIBNAME}_H}" MATCHES "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.[0-9]+\\.[0-9]+\\.([0-9]+).*$")
      set(${LIBNAME}_VERSION_TWEAK "${CMAKE_MATCH_1}" ${ARGN} PARENT_SCOPE)
    endif()
    if(${LIBNAME}_VERSION_TWEAK)
      set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_STRING}.${${LIBNAME}_VERSION_TWEAK}" ${ARGN} PARENT_SCOPE)
    else()
      set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_STRING}" ${ARGN} PARENT_SCOPE)
    endif()
  endif()
endfunction()

########################################################################################################
# An option that the user can select. Can accept condition to control when option is available for user.
# Usage:
#   caffe_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
function(caffe_option variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()
  unset(__varname)
  if("${__condition}" STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if("${__value}" MATCHES ";")
      if(${__value})
        option(${variable} "${description}" ON)
      else()
        option(${variable} "${description}" OFF)
      endif()
    elseif(DEFINED ${__value})
      if(${__value})
        option(${variable} "${description}" ON)
      else()
        option(${variable} "${description}" OFF)
      endif()
    else()
      option(${variable} "${description}" ${__value})
    endif()
  else()
    unset(${variable} CACHE)
  endif()
endfunction()

################################################################################################
# Utility macro for comparing two lists. Used for CMake debugging purposes
# Usage:
#   caffe_compare_lists(<list_variable> <list2_variable> [description])
function(caffe_compare_lists list1 list2 desc)
  set(__list1 ${${list1}})
  set(__list2 ${${list2}})
  list(SORT __list1)
  list(SORT __list2)
  list(LENGTH __list1 __len1)
  list(LENGTH __list2 __len2)

  if(NOT ${__len1} EQUAL ${__len2})
    message(FATAL_ERROR "Lists are not equal. ${__len1} != ${__len2}. ${desc}")
  endif()

  foreach(__i RANGE 1 ${__len1})
    math(EXPR __index "${__i}- 1")
    list(GET __list1 ${__index} __item1)
    list(GET __list2 ${__index} __item2)
    if(NOT ${__item1} STREQUAL ${__item2})
      message(FATAL_ERROR "Lists are not equal. Differ at element ${__index}. ${desc}")
    endif()
  endforeach()
endfunction()

################################################################################################
# Command for disabling warnings for different platforms (see below for gcc and VisualStudio)
# Usage:
#   caffe_warnings_disable(<CMAKE_[C|CXX]_FLAGS[_CONFIGURATION]> -Wshadow /wd4996 ..,)
macro(caffe_warnings_disable)
  set(_flag_vars "")
  set(_msvc_warnings "")
  set(_gxx_warnings "")

  foreach(arg ${ARGN})
    if(arg MATCHES "^CMAKE_")
      list(APPEND _flag_vars ${arg})
    elseif(arg MATCHES "^/wd")
      list(APPEND _msvc_warnings ${arg})
    elseif(arg MATCHES "^-W")
      list(APPEND _gxx_warnings ${arg})
    endif()
  endforeach()

  if(NOT _flag_vars)
    set(_flag_vars CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()

  if(MSVC AND _msvc_warnings)
    foreach(var ${_flag_vars})
      foreach(warning ${_msvc_warnings})
        set(${var} "${${var}} ${warning}")
      endforeach()
    endforeach()
  elseif((CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX) AND _gxx_warnings)
    foreach(var ${_flag_vars})
      foreach(warning ${_gxx_warnings})
        if(NOT warning MATCHES "^-Wno-")
          string(REPLACE "${warning}" "" ${var} "${${var}}")
          string(REPLACE "-W" "-Wno-" warning "${warning}")
        endif()
        set(${var} "${${var}} ${warning}")
      endforeach()
    endforeach()
  endif()
  caffe_clear_vars(_flag_vars _msvc_warnings _gxx_warnings)
endmacro()

################################################################################################
# Helper function get current definitions
# Usage:
#   caffe_get_current_definitions(<definitions_variable>)
function(caffe_get_current_definitions definitions_var)
  get_property(current_definitions DIRECTORY PROPERTY COMPILE_DEFINITIONS)
  set(result "")

  foreach(d ${current_definitions})
    list(APPEND result -D${d})
  endforeach()

  caffe_list_unique(result)
  set(${definitions_var} ${result} PARENT_SCOPE)
endfunction()

################################################################################################
# Helper function get current includes/definitions
# Usage:
#   caffe_get_current_cflags(<cflagslist_variable>)
function(caffe_get_current_cflags cflags_var)
  get_property(current_includes DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  caffe_convert_absolute_paths(current_includes)
  caffe_get_current_definitions(cflags)

  foreach(i ${current_includes})
    list(APPEND cflags "-I${i}")
  endforeach()

  caffe_list_unique(cflags)
  set(${cflags_var} ${cflags} PARENT_SCOPE)
endfunction()

################################################################################################
# Helper function to parse current linker libs into link directories, libflags and osx frameworks
# Usage:
#   caffe_parse_linker_libs(<Caffe_LINKER_LIBS_var> <directories_var> <libflags_var> <frameworks_var>)
function(caffe_parse_linker_libs Caffe_LINKER_LIBS_variable folders_var flags_var frameworks_var)

  set(__unspec "")
  set(__debug "")
  set(__optimized "")
  set(__framework "")
  set(__varname "__unspec")

  # split libs into debug, optimized, unspecified and frameworks
  foreach(list_elem ${${Caffe_LINKER_LIBS_variable}})
    if(list_elem STREQUAL "debug")
      set(__varname "__debug")
    elseif(list_elem STREQUAL "optimized")
      set(__varname "__optimized")
    elseif(list_elem MATCHES "^-framework[ \t]+([^ \t].*)")
      list(APPEND __framework -framework ${CMAKE_MATCH_1})
    else()
      list(APPEND ${__varname} ${list_elem})
      set(__varname "__unspec")
    endif()
  endforeach()

  # attach debug or optimized libs to unspecified according to current configuration
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(__libs ${__unspec} ${__debug})
  else()
    set(__libs ${__unspec} ${__optimized})
  endif()

  set(libflags "")
  set(folders "")

  # convert linker libraries list to link flags
  foreach(lib ${__libs})
    if(TARGET ${lib})
      list(APPEND folders $<TARGET_LINKER_FILE_DIR:${lib}>)
      list(APPEND libflags -l${lib})
    elseif(lib MATCHES "^-l.*")
      list(APPEND libflags ${lib})
    elseif(IS_ABSOLUTE ${lib})
      get_filename_component(folder  ${lib} PATH)
      get_filename_component(filename ${lib} NAME)
      string(REGEX REPLACE "\\.[^.]*$" "" filename_without_shortest_ext ${filename})

      string(REGEX MATCH "^lib(.*)" __match ${filename_without_shortest_ext})
      list(APPEND libflags -l${CMAKE_MATCH_1})
      list(APPEND folders    ${folder})
    else()
      message(FATAL_ERROR "Logic error. Need to update cmake script")
    endif()
  endforeach()

  caffe_list_unique(libflags folders)

  set(${folders_var} ${folders} PARENT_SCOPE)
  set(${flags_var} ${libflags} PARENT_SCOPE)
  set(${frameworks_var} ${__framework} PARENT_SCOPE)
endfunction()

################################################################################################
# Helper function to detect Darwin version, i.e. 10.8, 10.9, 10.10, ....
# Usage:
#   caffe_detect_darwin_version(<version_variable>)
function(caffe_detect_darwin_version output_var)
  if(APPLE)
    execute_process(COMMAND /usr/bin/sw_vers -productVersion
                    RESULT_VARIABLE __sw_vers OUTPUT_VARIABLE __sw_vers_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(${output_var} ${__sw_vers_out} PARENT_SCOPE)
  else()
    set(${output_var} "" PARENT_SCOPE)
  endif()
endfunction()
