# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__COMPILER_LLVM_INTEL)
  return()
endif()
set(__COMPILER_LLVM_INTEL 1)

include(Compiler/CMakeCommonCompilerMacros)

set(__pch_header_C "c-header")
set(__pch_header_CXX "c++-header")
set(__pch_header_OBJC "objective-c-header")
set(__pch_header_OBJCXX "objective-c++-header")

# Variables that are common across front-end variants
macro(__compiler_intel_llvm_common lang)
  set(_CMAKE_${lang}_IPO_SUPPORTED_BY_CMAKE YES)
  set(_CMAKE_${lang}_IPO_MAY_BE_SUPPORTED_BY_COMPILER YES)
  set(CMAKE_${lang}_ARCHIVE_CREATE_IPO "\"${CMAKE_${lang}_COMPILER_AR}\" qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_APPEND_IPO "\"${CMAKE_${lang}_COMPILER_AR}\" q <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_FINISH_IPO "\"${CMAKE_${lang}_COMPILER_RANLIB}\" <TARGET>")

  set(CMAKE_${lang}_LINK_MODE DRIVER)
endmacro()

if(CMAKE_HOST_WIN32)
  # MSVC-like
  macro(__compiler_intel_llvm lang)
    if("x${lang}" STREQUAL "xFortran")
      set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-warn:errors")
    else()
      set(CMAKE_${lang}_COMPILE_OPTIONS_INVALID_PCH -Winvalid-pch)
      set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-WX")
      if(CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL "2021.4")
        set(CMAKE_INCLUDE_SYSTEM_FLAG_${lang} "-external:I")
        if(CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL "2022.2")
          set(CMAKE_INCLUDE_SYSTEM_FLAG_${lang}_WARNING "-external:W0 ")
        endif()
      endif()
    endif()
    __compiler_intel_llvm_common(${lang})
    set(CMAKE_${lang}_COMPILE_OPTIONS_IPO "-Qipo")
    set(CMAKE_${lang}_LINK_OPTIONS_IPO "-Qipo")
  endmacro()
else()
  # GNU-like
  macro(__compiler_intel_llvm lang)
    set(CMAKE_${lang}_VERBOSE_FLAG "-v")

    string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
    string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -g")
    if(CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL 2023.0.0)
      if("x${lang}" STREQUAL "xFortran")
        string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -diag-disable:10440")
      else()
        string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -Rno-debug-disables-optimization")
      endif()
    endif()
    string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -Os")
    string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -O3")
    string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -O2 -g")

    set(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER YES)
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIE")
    set(CMAKE_${lang}_LINK_OPTIONS_PIE ${CMAKE_${lang}_COMPILE_OPTIONS_PIE} "-pie")
    set(CMAKE_${lang}_LINK_OPTIONS_NO_PIE "-no-pie")

    set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "-fPIC")
    set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared")

    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Wl,")
    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP ",")

    set(CMAKE_${lang}_LINK_MODE DRIVER)

    # distcc does not transform -o to -MT when invoking the preprocessor
    # internally, as it ought to.  Work around this bug by setting -MT here
    # even though it isn't strictly necessary.
    set(CMAKE_DEPFILE_FLAGS_${lang} "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")

    set(CMAKE_INCLUDE_SYSTEM_FLAG_${lang} "-isystem ")
    set(CMAKE_${lang}_COMPILE_OPTIONS_VISIBILITY "-fvisibility=")
    set(CMAKE_${lang}_COMPILE_OPTIONS_TARGET "--target=")
    set(CMAKE_${lang}_COMPILE_OPTIONS_SYSROOT "--sysroot=")
    set(CMAKE_${lang}_COMPILE_OPTIONS_EXTERNAL_TOOLCHAIN "--gcc-toolchain=")
    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Xlinker" " ")
    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP)

    __compiler_intel_llvm_common(${lang})
    set(CMAKE_${lang}_COMPILE_OPTIONS_IPO "-ipo")
    set(CMAKE_${lang}_LINK_OPTIONS_IPO "-ipo")

    set(CMAKE_${lang}_CREATE_PREPROCESSED_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E <SOURCE> > <PREPROCESSED_SOURCE>")
    set(CMAKE_${lang}_CREATE_ASSEMBLY_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -S <SOURCE> -o <ASSEMBLY_SOURCE>")

    if("${lang}" STREQUAL "CXX")
      set(CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "${CMAKE_${lang}_COMPILER}")
      if(CMAKE_${lang}_COMPILER_ARG1)
        separate_arguments(_COMPILER_ARGS NATIVE_COMMAND "${CMAKE_${lang}_COMPILER_ARG1}")
        list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND ${_COMPILER_ARGS})
        unset(_COMPILER_ARGS)
      endif()
      list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "-w" "-dM" "-E" "${CMAKE_ROOT}/Modules/CMakeCXXCompilerABI.cpp")
      if(CMAKE_${lang}_COMPILER_TARGET)
        list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "--target=${CMAKE_${lang}_COMPILER_TARGET}")
      endif()
    endif()

    if("x${lang}" STREQUAL "xFortran")
      set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-warn" "errors")
    else()
      # Precompile Headers
      set(CMAKE_PCH_EXTENSION .pch)
      set(CMAKE_PCH_PROLOGUE "#pragma clang system_header")
      set(CMAKE_${lang}_COMPILE_OPTIONS_INSTANTIATE_TEMPLATES_PCH -fpch-instantiate-templates)
      set(CMAKE_${lang}_COMPILE_OPTIONS_INVALID_PCH -Winvalid-pch)
      set(CMAKE_${lang}_COMPILE_OPTIONS_USE_PCH -Xclang -include-pch -Xclang <PCH_FILE> -Xclang -include -Xclang <PCH_HEADER>)
      set(CMAKE_${lang}_COMPILE_OPTIONS_CREATE_PCH -Xclang -emit-pch -Xclang -include -Xclang <PCH_HEADER> -x ${__pch_header_${lang}})

      # COMPILE_WARNING_AS_ERROR
      set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-Werror")
    endif()
  endmacro()
endif()
