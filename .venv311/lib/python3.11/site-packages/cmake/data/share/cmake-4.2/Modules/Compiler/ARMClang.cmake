if(_ARMClang_CMAKE_LOADED)
  return()
endif()

# This file requires CMAKE_LINKER and CMAKE_AR set by CMakeFindBinUtils.cmake.
if(NOT (DEFINED CMAKE_LINKER AND DEFINED CMAKE_AR))
  return()
endif()

set(_ARMClang_CMAKE_LOADED TRUE)

# Save the CMP0123 setting in a variable used both below and by try_compile.
cmake_policy(GET CMP0123 CMAKE_ARMClang_CMP0123)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

if (CMAKE_LINKER MATCHES "armlink")
  set(__CMAKE_ARMClang_USING_armlink TRUE)
  set(CMAKE_LIBRARY_PATH_FLAG "--userlibpath=")
else()
  set(__CMAKE_ARMClang_USING_armlink FALSE)
endif()

# get compiler supported cpu list
function(__armclang_set_processor_list lang out_var)
  execute_process(COMMAND "${CMAKE_${lang}_COMPILER}" --target=${CMAKE_${lang}_COMPILER_TARGET} -mcpu=list
    OUTPUT_VARIABLE processor_list
    ERROR_VARIABLE processor_list)
  string(REGEX MATCHALL "-mcpu=([^ \n]*)" processor_list "${processor_list}")
  string(REGEX REPLACE "-mcpu=" "" processor_list "${processor_list}")
  set(${out_var} "${processor_list}" PARENT_SCOPE)
endfunction()

# check processor is in list
function(__armclang_check_processor processor list out_var)
  string(TOLOWER "${processor}" processor)
  if(processor IN_LIST list)
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

# get compiler supported arch list
function(__armclang_set_arch_list lang out_var)
  execute_process(COMMAND "${CMAKE_${lang}_COMPILER}" --target=${CMAKE_${lang}_COMPILER_TARGET} -march=list
    OUTPUT_VARIABLE arch_list
    ERROR_VARIABLE arch_list)
  string(REGEX MATCHALL "-march=([^ \n]*)" arch_list "${arch_list}")
  string(REGEX REPLACE "-march=" "" arch_list "${arch_list}")
  set(${out_var} "${arch_list}" PARENT_SCOPE)
endfunction()

# get linker supported cpu list
function(__armlink_set_cpu_list lang out_var)
  if(__CMAKE_ARMClang_USING_armlink)
    set(__linker_wrapper_flags "")
  else()
    set(__linker_wrapper_flags --target=${CMAKE_${lang}_COMPILER_TARGET} -Xlinker)
  endif()

  execute_process(COMMAND "${CMAKE_LINKER}" ${__linker_wrapper_flags} --cpu=list
    OUTPUT_VARIABLE cpu_list
    ERROR_VARIABLE cpu_list)
  string(REGEX MATCHALL "--cpu=([^ \n]*)" cpu_list "${cpu_list}")
  string(REGEX REPLACE "--cpu=" "" cpu_list "${cpu_list}")
  set(${out_var} "${cpu_list}" PARENT_SCOPE)
endfunction()

macro(__compiler_armclang lang)
  if(NOT CMAKE_${lang}_COMPILER_TARGET)
    set(CMAKE_${lang}_COMPILER_TARGET arm-arm-none-eabi)
  endif()
  if(NOT CMAKE_${lang}_COMPILER_PROCESSOR_LIST)
    __armclang_set_processor_list(${lang} CMAKE_${lang}_COMPILER_PROCESSOR_LIST)
  endif()
  if(NOT CMAKE_${lang}_COMPILER_ARCH_LIST)
    __armclang_set_arch_list(${lang} CMAKE_${lang}_COMPILER_ARCH_LIST)
  endif()

  # CMAKE_SYSTEM_PROCESSOR and CMAKE_SYSTEM_ARCH are not sufficient because they provide no
  # information of additional CPU features needed in `-mcpu=<name>[+[no]<feature>+...]`.
  # The automatic setting of compile and link options is deprecated and projects should specify their own.
  cmake_policy(GET CMP0123 policy_CMP0123)
  if(NOT "x${CMAKE_ARMClang_CMP0123}x" STREQUAL "xNEWx")
    if(NOT "x${CMAKE_ARMClang_CMP0123}x" STREQUAL "xOLDx")
      cmake_policy(GET_WARNING CMP0123 _cmp0123_warning)
      message(AUTHOR_WARNING
        "${_cmp0123_warning}\n"
        "For compatibility, CMake will automatically add cpu/arch flags based "
        "on the CMAKE_SYSTEM_PROCESSOR and/or CMAKE_SYSTEM_ARCH variables."
        )
    endif()

    if(NOT CMAKE_SYSTEM_PROCESSOR AND NOT CMAKE_SYSTEM_ARCH)
      message(FATAL_ERROR "  CMAKE_SYSTEM_PROCESSOR or CMAKE_SYSTEM_ARCH must be set for ARMClang\n"
        "  Supported processor: ${CMAKE_${lang}_COMPILER_PROCESSOR_LIST}\n"
        "  Supported Architecture: ${CMAKE_${lang}_COMPILER_ARCH_LIST}")
    else()
      __armclang_check_processor("${CMAKE_SYSTEM_ARCH}" "${CMAKE_${lang}_COMPILER_ARCH_LIST}" _CMAKE_${lang}_CHECK_ARCH_RESULT)
      if( _CMAKE_${lang}_CHECK_ARCH_RESULT)
        string(APPEND CMAKE_${lang}_FLAGS_INIT " -march=${CMAKE_SYSTEM_ARCH}")
        set(__march_flag_set TRUE)
      endif()
      __armclang_check_processor("${CMAKE_SYSTEM_PROCESSOR}" "${CMAKE_${lang}_COMPILER_PROCESSOR_LIST}" _CMAKE_${lang}_CHECK_PROCESSOR_RESULT)
      if(_CMAKE_${lang}_CHECK_PROCESSOR_RESULT)
        string(APPEND CMAKE_${lang}_FLAGS_INIT " -mcpu=${CMAKE_SYSTEM_PROCESSOR}")
        set(__mcpu_flag_set TRUE)
      endif()
      if(NOT __march_flag_set AND NOT __mcpu_flag_set)
        message(FATAL_ERROR "At least one of the variables CMAKE_SYSTEM_PROCESSOR or CMAKE_SYSTEM_ARCH must be set for ARMClang\n"
                            "Supported processor: ${CMAKE_${lang}_COMPILER_PROCESSOR_LIST}\n"
                            "  Supported Architecture: ${CMAKE_${lang}_COMPILER_ARCH_LIST}")
      endif()
      unset(_CMAKE_${lang}_CHECK_PROCESSOR_RESULT)
      unset(_CMAKE_${lang}_CHECK_ARCH_RESULT)
    endif()

    #check if CMAKE_SYSTEM_PROCESSOR belongs to supported cpu list for armlink
    __armlink_set_cpu_list( ${lang} CMAKE_LINKER_CPU_LIST)
    list(TRANSFORM CMAKE_LINKER_CPU_LIST TOLOWER)
    __armclang_check_processor("${CMAKE_SYSTEM_PROCESSOR}" "${CMAKE_LINKER_CPU_LIST}" _CMAKE_CHECK_LINK_CPU_RESULT)
    if(_CMAKE_CHECK_LINK_CPU_RESULT)
      string(APPEND CMAKE_${lang}_LINK_FLAGS " --cpu=${CMAKE_SYSTEM_PROCESSOR}")
    endif()

  endif()

  if(__CMAKE_ARMClang_USING_armlink)
    unset(CMAKE_${lang}_LINKER_WRAPPER_FLAG)
    set(__CMAKE_ARMClang_USING_armlink_WRAPPER "")
  else()
    set(__CMAKE_ARMClang_USING_armlink_WRAPPER "-Xlinker")
  endif()
  set(CMAKE_${lang}_LINK_EXECUTABLE "<CMAKE_LINKER> <LINK_FLAGS> <LINK_LIBRARIES> <OBJECTS> -o <TARGET> ${__CMAKE_ARMClang_USING_armlink_WRAPPER}")
  set(CMAKE_${lang}_CREATE_STATIC_LIBRARY  "<CMAKE_AR> --create -cr <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_CREATE         "<CMAKE_AR> --create -cr <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "${__CMAKE_ARMClang_USING_armlink_WRAPPER} --via=")
  set(CMAKE_${lang}_OUTPUT_EXTENSION ".o")
  set(CMAKE_${lang}_OUTPUT_EXTENSION_REPLACE 1)
endmacro()
