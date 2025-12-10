# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


include(Compiler/GNU)

macro(__compiler_qcc lang)
  __compiler_gnu(${lang})

  # http://www.qnx.com/developers/docs/6.4.0/neutrino/utilities/q/qcc.html#examples
  set(CMAKE_${lang}_COMPILE_OPTIONS_TARGET "-V")

  set(CMAKE_PREFIX_LIBRARY_ARCHITECTURE "ON")

  set(CMAKE_${lang}_COMPILE_OPTIONS_SYSROOT "-Wc,-isysroot,")
  set(CMAKE_INCLUDE_SYSTEM_FLAG_${lang} "-isystem ")
  set(CMAKE_DEPFILE_FLAGS_${lang} "-Wp,-MD,<DEP_FILE> -Wp,-MT,<DEP_TARGET> -Wp,-MF,<DEP_FILE>")

  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Wl,")
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP ",")

  if("${lang}" STREQUAL "CXX")
    set(CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "${CMAKE_${lang}_COMPILER}")
    if(CMAKE_${lang}_COMPILER_ARG1)
      separate_arguments(_COMPILER_ARGS NATIVE_COMMAND "${CMAKE_${lang}_COMPILER_ARG1}")
      list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND ${_COMPILER_ARGS})
      unset(_COMPILER_ARGS)
    endif()
    list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "-Wp,-dM" "-E" "-c" "${CMAKE_ROOT}/Modules/CMakeCXXCompilerABI.cpp")
  endif()
endmacro()
