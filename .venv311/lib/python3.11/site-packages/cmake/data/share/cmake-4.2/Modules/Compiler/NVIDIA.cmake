
# This module is shared by multiple languages; use include blocker.
if(__COMPILER_NVIDIA)
  return()
endif()
set(__COMPILER_NVIDIA 1)

include(Compiler/CMakeCommonCompilerMacros)

macro(__compiler_nvidia_cxx_standards lang)
  if("x${CMAKE_${lang}_SIMULATE_ID}" STREQUAL "xMSVC")
    # MSVC requires c++14 as the minimum level
    set(CMAKE_${lang}03_STANDARD_COMPILE_OPTION "")
    set(CMAKE_${lang}03_EXTENSION_COMPILE_OPTION "")

    # MSVC requires c++14 as the minimum level
    set(CMAKE_${lang}11_STANDARD_COMPILE_OPTION "")
    set(CMAKE_${lang}11_EXTENSION_COMPILE_OPTION "")

    set(CMAKE_${lang}_STANDARD_LATEST 11)

    if (NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 9.0)
      if(CMAKE_${lang}_SIMULATE_VERSION VERSION_GREATER_EQUAL 19.10.25017)
        set(CMAKE_${lang}14_STANDARD_COMPILE_OPTION "-std=c++14")
        set(CMAKE_${lang}14_EXTENSION_COMPILE_OPTION "-std=c++14")
      else()
        set(CMAKE_${lang}14_STANDARD_COMPILE_OPTION "")
        set(CMAKE_${lang}14_EXTENSION_COMPILE_OPTION "")
      endif()

      set(CMAKE_${lang}_STANDARD_LATEST 14)
    endif()

    if (NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 11.0)
      if(CMAKE_${lang}_SIMULATE_VERSION VERSION_GREATER_EQUAL 19.11.25505)
        set(CMAKE_${lang}17_STANDARD_COMPILE_OPTION "-std=c++17")
        set(CMAKE_${lang}17_EXTENSION_COMPILE_OPTION "-std=c++17")
        set(CMAKE_${lang}_STANDARD_LATEST 17)
      endif()
    endif()

    if (NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 12.0)
      if(CMAKE_${lang}_SIMULATE_VERSION VERSION_GREATER_EQUAL 19.11.25505)
        set(CMAKE_${lang}20_STANDARD_COMPILE_OPTION "-std=c++20")
        set(CMAKE_${lang}20_EXTENSION_COMPILE_OPTION "-std=c++20")
        set(CMAKE_${lang}_STANDARD_LATEST 20)
      endif()
    endif()
  else()
    set(CMAKE_${lang}03_STANDARD_COMPILE_OPTION "")
    set(CMAKE_${lang}03_EXTENSION_COMPILE_OPTION "")

    set(CMAKE_${lang}11_STANDARD_COMPILE_OPTION "-std=c++11")
    set(CMAKE_${lang}11_EXTENSION_COMPILE_OPTION "-std=c++11")

    set(CMAKE_${lang}_STANDARD_LATEST 11)

    if (NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 9.0)
      set(CMAKE_${lang}03_STANDARD_COMPILE_OPTION "-std=c++03")
      set(CMAKE_${lang}03_EXTENSION_COMPILE_OPTION "-std=c++03")
      set(CMAKE_${lang}14_STANDARD_COMPILE_OPTION "-std=c++14")
      set(CMAKE_${lang}14_EXTENSION_COMPILE_OPTION "-std=c++14")

      set(CMAKE_${lang}_STANDARD_LATEST 14)
    endif()

    if (NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 11.0)
      set(CMAKE_${lang}17_STANDARD_COMPILE_OPTION "-std=c++17")
      set(CMAKE_${lang}17_EXTENSION_COMPILE_OPTION "-std=c++17")
      set(CMAKE_${lang}_STANDARD_LATEST 17)
    endif()

    if (NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 12.0)
      set(CMAKE_${lang}20_STANDARD_COMPILE_OPTION "-std=c++20")
      set(CMAKE_${lang}20_EXTENSION_COMPILE_OPTION "-std=c++20")
      set(CMAKE_${lang}_STANDARD_LATEST 20)
    endif()
  endif()

  __compiler_check_default_language_standard(${lang} 6.0 03)
endmacro()

macro(__compiler_nvidia_cuda_flags lang)
  set(CMAKE_${lang}_VERBOSE_FLAG "-v")
  set(CMAKE_${lang}_VERBOSE_COMPILE_FLAG "-Xcompiler=-v")
  set(_CMAKE_COMPILE_AS_${lang}_FLAG "-x cu")

  if (CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2.89)
    # The -forward-unknown-to-host-compiler flag was only
    # added to nvcc in 10.2 so before that we had no good
    # way to invoke the NVCC compiler and propagate unknown
    # flags such as -pthread to the host compiler
    set(_CMAKE_${lang}_EXTRA_FLAGS "-forward-unknown-to-host-compiler")
  else()
    set(_CMAKE_${lang}_EXTRA_FLAGS "")
  endif()

  if(CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL "8.0.0")
    set(_CMAKE_${lang}_EXTRA_DEVICE_LINK_FLAGS "-Wno-deprecated-gpu-targets")
  else()
    set(_CMAKE_${lang}_EXTRA_DEVICE_LINK_FLAGS "")
  endif()

  if(CMAKE_${lang}_HOST_COMPILER AND NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    string(APPEND _CMAKE_${lang}_EXTRA_FLAGS " -ccbin=<CMAKE_${lang}_HOST_COMPILER>")
  endif()

  if (CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2.89)
    # Starting in 10.2, nvcc supported treating all warnings as errors
    set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-Werror" "all-warnings")
  endif()

  set(CMAKE_${lang}_DEPFILE_FORMAT gcc)
  if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
      AND CMAKE_GENERATOR MATCHES "Makefiles|WMake")
    set(CMAKE_${lang}_DEPENDS_USE_COMPILER TRUE)
  endif()

  if (CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2.89)
    # The -MD flag was only added to nvcc in 10.2 so
    # before that we had to invoke the compiler twice
    # to get header dependency information
    set(CMAKE_DEPFILE_FLAGS_${lang} "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
  else()
    set(CMAKE_${lang}_DEPENDS_EXTRA_COMMANDS "<CMAKE_${lang}_COMPILER> ${_CMAKE_${lang}_EXTRA_FLAGS} <DEFINES> <INCLUDES> <FLAGS> ${_CMAKE_COMPILE_AS_${lang}_FLAG} -M <SOURCE> -MT <OBJECT> -o <DEP_FILE>")
  endif()

  if(CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2)
    set(_CMAKE_${lang}_IPO_SUPPORTED_BY_CMAKE YES)
    set(_CMAKE_${lang}_IPO_MAY_BE_SUPPORTED_BY_COMPILER YES)
  endif()

  if(NOT "x${CMAKE_${lang}_SIMULATE_ID}" STREQUAL "xMSVC")
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIE -Xcompiler=-fPIE)
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIC -Xcompiler=-fPIC)
    set(CMAKE_${lang}_COMPILE_OPTIONS_VISIBILITY -Xcompiler=-fvisibility=)
    # CMAKE_SHARED_LIBRARY_${lang}_FLAGS is sent to the host linker so we
    # don't need to forward it through nvcc.
    set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS -fPIC)
    string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
    string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -g")
    string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -O3 -DNDEBUG")
    string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -O1 -DNDEBUG")
    string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -O2 -g -DNDEBUG")
  endif()

  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS -shared)
  set(CMAKE_INCLUDE_SYSTEM_FLAG_${lang} "-isystem ")

  if (CMAKE_${lang}_SIMULATE_ID STREQUAL "GNU")
    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Wl,")
    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP ",")
  elseif(CMAKE_${lang}_SIMULATE_ID STREQUAL "Clang")
    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Xlinker" " ")
    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP)
  endif()

  if (CMAKE_${lang}_SIMULATE_ID STREQUAL "MSVC")
    set(CMAKE_${lang}_LINK_MODE LINKER)
  else()
    set(CMAKE_${lang}_LINK_MODE DRIVER)
  endif()

  set(CMAKE_${lang}_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC  "cudadevrt;cudart_static")
  set(CMAKE_${lang}_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED  "cudadevrt;cudart")
  set(CMAKE_${lang}_RUNTIME_LIBRARY_LINK_OPTIONS_NONE    "")

  if(UNIX AND NOT (CMAKE_SYSTEM_NAME STREQUAL "QNX"))
    list(APPEND CMAKE_${lang}_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC "rt" "pthread" "dl")
  endif()

  if (CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL "9.0")
    set(CMAKE_${lang}_RESPONSE_FILE_DEVICE_LINK_FLAG "--options-file ")
    set(CMAKE_${lang}_RESPONSE_FILE_FLAG "--options-file ")
  endif()

  if (CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
    set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_INCLUDES 1)
    set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_LIBRARIES 1)
    set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_OBJECTS 1)
  else()
    set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_INCLUDES 0)
    set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
    set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_OBJECTS 0)
  endif()
endmacro()
