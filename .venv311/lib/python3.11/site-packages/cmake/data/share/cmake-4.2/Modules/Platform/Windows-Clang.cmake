# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__WINDOWS_CLANG)
  return()
endif()
set(__WINDOWS_CLANG 1)

set(__pch_header_C "c-header")
set(__pch_header_CXX "c++-header")
set(__pch_header_OBJC "objective-c-header")
set(__pch_header_OBJCXX "objective-c++-header")

macro(__windows_compiler_clang_gnu lang)
  set(CMAKE_LIBRARY_PATH_FLAG "-L")
  set(CMAKE_LINK_LIBRARY_FLAG "-l")

  set(CMAKE_IMPORT_LIBRARY_PREFIX "")
  set(CMAKE_SHARED_LIBRARY_PREFIX "")
  set(CMAKE_SHARED_MODULE_PREFIX  "")
  set(CMAKE_STATIC_LIBRARY_PREFIX "")
  set(CMAKE_EXECUTABLE_SUFFIX     ".exe")
  set(CMAKE_IMPORT_LIBRARY_SUFFIX ".lib")
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
  set(CMAKE_SHARED_MODULE_SUFFIX  ".dll")
  set(CMAKE_STATIC_LIBRARY_SUFFIX ".lib")
  if(NOT "${lang}" STREQUAL "ASM")
    set(CMAKE_DEPFILE_FLAGS_${lang} "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
  endif()

  set(CMAKE_FIND_LIBRARY_PREFIXES "lib" "")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll.a" ".a" ".lib")
  set(CMAKE_SUPPORT_WINDOWS_EXPORT_ALL_SYMBOLS 1)
  set(CMAKE_${lang}_LINK_DEF_FILE_FLAG "-Xlinker /DEF:")
  set(CMAKE_LINK_DEF_FILE_FLAG "${CMAKE_${lang}_LINK_DEF_FILE_FLAG}")

  set(CMAKE_${lang}_COMPILE_OPTIONS_SYSROOT "--sysroot=")

  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Xlinker" " ")
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP)

  set(CMAKE_${lang}_LINK_MODE DRIVER)

  set(CMAKE_${lang}_LINKER_MANIFEST_FLAG " -Xlinker /MANIFESTINPUT:")
  set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-Werror")

  if("${CMAKE_${lang}_SIMULATE_VERSION}" MATCHES "^([0-9]+)\\.([0-9]+)")
    math(EXPR MSVC_VERSION "${CMAKE_MATCH_1}*100 + ${CMAKE_MATCH_2}")
  endif()

  set(CMAKE_${lang}_VERBOSE_LINK_FLAG "-v")
  # No -fPIC on Windows
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "")
  set(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER NO)
  set(CMAKE_${lang}_LINK_OPTIONS_PIE "")
  set(CMAKE_${lang}_LINK_OPTIONS_NO_PIE "")
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "")
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared")
  set(CMAKE_${lang}_SHARED_LIBRARY_COMPILE_DEFINITIONS "_WINDLL")

  # linker selection
  set(CMAKE_${lang}_USING_LINKER_DEFAULT "-fuse-ld=lld-link")
  set(CMAKE_${lang}_USING_LINKER_SYSTEM "-fuse-ld=link")
  set(CMAKE_${lang}_USING_LINKER_LLD "-fuse-ld=lld-link")
  set(CMAKE_${lang}_USING_LINKER_MSVC "-fuse-ld=link")

  set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_OBJECTS 1)
  set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_LIBRARIES 1)
  set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_INCLUDES 1)

  if(CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL 3.9)
    set(CMAKE_${lang}_COMPILE_OPTIONS_IPO "-flto=thin")
  else()
    set(CMAKE_${lang}_COMPILE_OPTIONS_IPO "-flto")
  endif()

  set(_CMAKE_${lang}_IPO_SUPPORTED_BY_CMAKE YES)
  set(_CMAKE_${lang}_IPO_MAY_BE_SUPPORTED_BY_COMPILER YES)
  set(CMAKE_${lang}_ARCHIVE_CREATE_IPO "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_APPEND_IPO "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_FINISH_IPO "<CMAKE_RANLIB> <TARGET>")

  # Create archiving rules to support large object file lists for static libraries.
  set(CMAKE_${lang}_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
  set(CMAKE_${lang}_CREATE_SHARED_LIBRARY
    "<CMAKE_${lang}_COMPILER> -nostartfiles -nostdlib <CMAKE_SHARED_LIBRARY_${lang}_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> -o <TARGET> ${CMAKE_GNULD_IMAGE_VERSION} -Xlinker /MANIFEST:EMBED -Xlinker /implib:<TARGET_IMPLIB> -Xlinker /pdb:<TARGET_PDB> -Xlinker /version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR> <OBJECTS> <LINK_LIBRARIES> <MANIFESTS>")
  set(CMAKE_${lang}_CREATE_SHARED_MODULE ${CMAKE_${lang}_CREATE_SHARED_LIBRARY})
  set(CMAKE_${lang}_LINK_EXECUTABLE
    "<CMAKE_${lang}_COMPILER> -nostartfiles -nostdlib <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> -Xlinker /MANIFEST:EMBED -Xlinker /implib:<TARGET_IMPLIB> -Xlinker /pdb:<TARGET_PDB> -Xlinker /version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR> ${CMAKE_GNULD_IMAGE_VERSION} <LINK_LIBRARIES> <MANIFESTS>")

  set(CMAKE_${lang}_CREATE_WIN32_EXE "-Xlinker /subsystem:windows")
  set(CMAKE_${lang}_CREATE_CONSOLE_EXE "-Xlinker /subsystem:console")

  if(NOT "${lang}" STREQUAL "ASM")
    set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded         -Xclang -flto-visibility-public-std -D_MT -Xclang --dependent-lib=libcmt)
    set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL      -D_DLL -D_MT -Xclang --dependent-lib=msvcrt)
    set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug    -D_DEBUG -Xclang -flto-visibility-public-std -D_MT -Xclang --dependent-lib=libcmtd)
    set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL -D_DEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrtd)

    if(CMAKE_MSVC_RUNTIME_LIBRARY_DEFAULT)
      set(_RTL_FLAGS "")
      set(_RTL_FLAGS_DEBUG "")
    else()
      set(_RTL_FLAGS_DEBUG " -D_DEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrtd")
      set(_RTL_FLAGS " -D_DLL -D_MT -Xclang --dependent-lib=msvcrt")
    endif()

    if(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT_DEFAULT)
      set(_DBG_FLAGS "")
    else()
      set(_DBG_FLAGS " -g -Xclang -gcodeview")
    endif()

    string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -O0${_DBG_FLAGS}${_RTL_FLAGS_DEBUG}")
    string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -Os -DNDEBUG${_RTL_FLAGS}")
    string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -O3 -DNDEBUG${_RTL_FLAGS}")
    string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -O2 -DNDEBUG${_DBG_FLAGS}${_RTL_FLAGS}")

    # clang-cl accepts -RTC* flags but ignores them.  Simulate this
    # with the GNU-like drivers by simply passing no flags at all.
    set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_RUNTIME_CHECKS_PossibleDataLoss      "")
    set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_RUNTIME_CHECKS_StackFrameErrorCheck  "")
    set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_RUNTIME_CHECKS_UninitializedVariable "")
    set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_RUNTIME_CHECKS_RTCsu                 "")

    set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_Embedded        -g -Xclang -gcodeview)
    #set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_ProgramDatabase) # not supported by Clang
    #set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_EditAndContinue) # not supported by Clang
  endif()
  set(CMAKE_INCLUDE_SYSTEM_FLAG_${lang} "-isystem ")
  set(CMAKE_${lang}_LINKER_SUPPORTS_PDB ON)

  set(CMAKE_PCH_EXTENSION .pch)
  set(CMAKE_PCH_PROLOGUE "#pragma clang system_header")
  set(CMAKE_${lang}_COMPILE_OPTIONS_USE_PCH -Xclang -include-pch -Xclang <PCH_FILE> -Xclang -include -Xclang <PCH_HEADER>)
  set(CMAKE_${lang}_COMPILE_OPTIONS_CREATE_PCH -Xclang -emit-pch -Xclang -include -Xclang <PCH_HEADER> -x ${__pch_header_${lang}})

  unset(_DBG_FLAGS)
  unset(_RTL_FLAGS)
  unset(_RTL_FLAGS_DEBUG)
  string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
  set(CMAKE_${lang}_STANDARD_LIBRARIES_INIT "-lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -loldnames")

  enable_language(RC)
endmacro()

macro(__enable_llvm_rc_preprocessing clang_option_prefix extra_pp_flags)
  # Feed the preprocessed rc file to llvm-rc
  if(CMAKE_RC_COMPILER_INIT MATCHES "llvm-rc" OR CMAKE_RC_COMPILER MATCHES "llvm-rc")
    if(DEFINED CMAKE_C_COMPILER_ID)
      set(CMAKE_RC_PREPROCESSOR CMAKE_C_COMPILER)
    elseif(DEFINED CMAKE_CXX_COMPILER_ID)
      set(CMAKE_RC_PREPROCESSOR CMAKE_CXX_COMPILER)
    endif()
    if(DEFINED CMAKE_RC_PREPROCESSOR)
      set(CMAKE_DEPFILE_FLAGS_RC "${clang_option_prefix}-MD ${clang_option_prefix}-MF ${clang_option_prefix}<DEP_FILE>")
      # llvm-rc runs preprocessor starting from LLVM-13, so we can run it directly instead of using "cmake_llvm_rc".
      # See https://reviews.llvm.org/D100755 for more details.
      if (CMAKE_GENERATOR MATCHES "FASTBuild")
        set(CMAKE_RC_COMPILE_OBJECT "<CMAKE_RC_COMPILER> <DEFINES> -I <SOURCE_DIR> <INCLUDES> <FLAGS> /fo<OBJECT> <SOURCE>")
      else()
        # The <FLAGS> are passed to the preprocess and the resource compiler to pick
        # up the eventual -D / -C options passed through the CMAKE_RC_FLAGS.
        set(CMAKE_RC_COMPILE_OBJECT "<CMAKE_COMMAND> -E cmake_llvm_rc <SOURCE> <OBJECT>.pp <${CMAKE_RC_PREPROCESSOR}> <DEFINES> -DRC_INVOKED <INCLUDES> <FLAGS> ${extra_pp_flags} -E -- <SOURCE> ++ <CMAKE_RC_COMPILER> <DEFINES> -I <SOURCE_DIR> <INCLUDES> <FLAGS> /fo <OBJECT> <OBJECT>.pp")
      endif()
      if(CMAKE_GENERATOR MATCHES "Ninja")
        set(CMAKE_NINJA_CMCLDEPS_RC 0)
        set(CMAKE_NINJA_DEP_TYPE_RC gcc)
      endif()
      unset(CMAKE_RC_PREPROCESSOR)
    endif()
  endif()
endmacro()

function(__verify_same_language_values variable langs)
  foreach(lang IN LISTS langs)
    list(APPEND __LANGUAGE_VALUES_${variable} ${CMAKE_${lang}_${variable}})
  endforeach()
  list(REMOVE_DUPLICATES __LANGUAGE_VALUES_${variable})
  list(LENGTH __LANGUAGE_VALUES_${variable} __NUM_VALUES)
  if(__NUM_VALUES GREATER 1)
    message(FATAL_ERROR ${ARGN})
  endif()
endfunction()

if("x${CMAKE_C_SIMULATE_ID}" STREQUAL "xMSVC"
    OR "x${CMAKE_CXX_SIMULATE_ID}" STREQUAL "xMSVC"
    OR "x${CMAKE_CUDA_SIMULATE_ID}" STREQUAL "xMSVC"
    OR "x${CMAKE_HIP_SIMULATE_ID}" STREQUAL "xMSVC")

  __verify_same_language_values(COMPILER_ID "C;CXX;HIP"
                                "The current configuration mixes Clang and MSVC or "
                                "some other CL compatible compiler tool. This is not supported. "
                                "Use either Clang or MSVC as the compiler for all of C, C++, and/or HIP.")

  __verify_same_language_values(COMPILER_FRONTEND_VARIANT "C;CXX;CUDA;HIP"
                                "The current configuration uses the Clang compiler "
                                "tool with mixed frontend variants, both the GNU and in MSVC CL "
                                "like variants. This is not supported. Use either clang/clang++ "
                                "or clang-cl as all C, C++, CUDA, and/or HIP compilers.")

  if(NOT CMAKE_RC_COMPILER_INIT)
    # Check if rc is already in the path
    # This may happen in cases where the user is already in a visual studio environment when CMake is invoked
    find_program(__RC_COMPILER_PATH NAMES rc)

    # Default to rc if it's available, otherwise fall back to llvm-rc
    if(__RC_COMPILER_PATH)
      set(CMAKE_RC_COMPILER_INIT rc)
    else()
      find_program(__RC_COMPILER_PATH NAMES llvm-rc)
      if(__RC_COMPILER_PATH)
        set(CMAKE_RC_COMPILER_INIT llvm-rc)
      endif()
    endif()

    unset(__RC_COMPILER_PATH CACHE)
  endif()

  if ( "x${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "xMSVC"
      OR "x${CMAKE_C_COMPILER_FRONTEND_VARIANT}" STREQUAL "xMSVC"
      OR "x${CMAKE_CUDA_COMPILER_FRONTEND_VARIANT}" STREQUAL "xMSVC"
      OR "x${CMAKE_HIP_COMPILER_FRONTEND_VARIANT}" STREQUAL "xMSVC")

    include(Platform/Windows-MSVC)
    # Set the clang option forwarding prefix for clang-cl usage in the llvm-rc processing stage
    __enable_llvm_rc_preprocessing("-clang:" "")
    macro(__windows_compiler_clang_base lang)
      set(_COMPILE_${lang} "${_COMPILE_${lang}_MSVC}")
      __windows_compiler_msvc(${lang})
      unset(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_EditAndContinue) # -ZI not supported by Clang
      set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-WX")
      set(CMAKE_INCLUDE_SYSTEM_FLAG_${lang} "-imsvc")

      set(CMAKE_${lang}_LINK_MODE LINKER)
    endmacro()
  else()
    cmake_policy(GET CMP0091 __WINDOWS_CLANG_CMP0091)
    if(__WINDOWS_CLANG_CMP0091 STREQUAL "NEW")
      set(CMAKE_MSVC_RUNTIME_LIBRARY_DEFAULT "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    else()
      set(CMAKE_MSVC_RUNTIME_LIBRARY_DEFAULT "")
    endif()
    unset(__WINDOWS_CLANG_CMP0091)

    cmake_policy(GET CMP0141 __WINDOWS_MSVC_CMP0141)
    if(__WINDOWS_MSVC_CMP0141 STREQUAL "NEW")
      set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT_DEFAULT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
    else()
      set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT_DEFAULT "")
    endif()
    unset(__WINDOWS_MSVC_CMP0141)

    cmake_policy(GET CMP0184 __WINDOWS_MSVC_CMP0184)
    if(__WINDOWS_MSVC_CMP0184 STREQUAL "NEW")
      set(CMAKE_MSVC_RUNTIME_CHECKS_DEFAULT "$<$<CONFIG:Debug>:StackFrameErrorCheck;UninitializedVariable>")
    else()
      set(CMAKE_MSVC_RUNTIME_CHECKS_DEFAULT "")
    endif()
    unset(__WINDOWS_MSVC_CMP0184)

    set(CMAKE_BUILD_TYPE_INIT Debug)

    __enable_llvm_rc_preprocessing("" "-x c")
    macro(__windows_compiler_clang_base lang)
      __windows_compiler_clang_gnu(${lang})
    endmacro()
  endif()

else()
  include(Platform/Windows-GNU)
  __enable_llvm_rc_preprocessing("" "-x c")
  macro(__windows_compiler_clang_base lang)
    __windows_compiler_gnu(${lang})

    set(CMAKE_${lang}_LINK_MODE DRIVER)
  endmacro()
endif()

macro(__windows_compiler_clang lang)
  if(CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 3.4.0)
    set(CMAKE_${lang}_COMPILE_OPTIONS_TARGET "-target ")
  else()
    set(CMAKE_${lang}_COMPILE_OPTIONS_TARGET "--target=")
  endif()
  __windows_compiler_clang_base(${lang})
endmacro()
