if("x${CMAKE_Fortran_SIMULATE_ID}" STREQUAL "xGNU")
  include(Platform/Windows-GNU)
  __windows_compiler_gnu(Fortran)
elseif("x${CMAKE_Fortran_SIMULATE_ID}" STREQUAL "xMSVC")
  include(Platform/Windows-MSVC)
  __windows_compiler_msvc(Fortran)

  if(CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL 18.0)
    set(_LLVMFlang_LINK_RUNTIME "")
    set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded         "-fms-runtime-lib=static")
    set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL      "-fms-runtime-lib=dll")
    set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug    "-fms-runtime-lib=static_dbg")
    set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL "-fms-runtime-lib=dll_dbg")
  else()
    # LLVMFlang < 18.0 does not have MSVC runtime library selection flags.
    # The official distribution's `Fortran*.lib` runtime libraries hard-code
    # use of msvcrt (MultiThreadedDLL), so we link to it ourselves.
    set(_LLVMFlang_LINK_RUNTIME "-defaultlib:msvcrt")
    set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded         "")
    set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL      "")
    set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug    "")
    set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL "")
  endif()

  # LLVMFlang, like Clang, does not provide all debug information format flags.
  # In order to provide easy integration with C and C++ projects that use the
  # other debug information formats, pretend to support them, and just do not
  # actually generate any debug information for Fortran.
  set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_Embedded        -g)
  set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_ProgramDatabase "") # not supported by LLVMFlang
  set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_EditAndContinue "") # not supported by LLVMFlang

  set(CMAKE_Fortran_COMPILE_OBJECT "<CMAKE_Fortran_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")

  if(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT_DEFAULT)
    set(_g "")
  else()
    set(_g " -g")
  endif()
  string(APPEND CMAKE_Fortran_FLAGS_DEBUG_INIT "${_g}")
  string(APPEND CMAKE_Fortran_FLAGS_RELEASE_INIT "")
  string(APPEND CMAKE_Fortran_FLAGS_RELWITHDEBINFO_INIT "${_g}")
  string(APPEND CMAKE_Fortran_FLAGS_MINSIZEREL_INIT "")
  unset(_g)

  # We link with lld-link.exe instead of the compiler driver, so explicitly
  # pass implicit link information previously detected from the compiler.
  set(_LLVMFlang_LINK_DIRS "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
  list(TRANSFORM _LLVMFlang_LINK_DIRS PREPEND "-libpath:\"")
  list(TRANSFORM _LLVMFlang_LINK_DIRS APPEND "\"")
  string(JOIN " " _LLVMFlang_LINK_DIRS ${_LLVMFlang_LINK_DIRS})
  string(JOIN " " _LLVMFlang_LINK_LIBS ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
  foreach(v IN ITEMS
      CMAKE_Fortran_LINK_EXECUTABLE
      CMAKE_Fortran_CREATE_SHARED_LIBRARY
      CMAKE_Fortran_CREATE_SHARED_MODULE
      )
    string(APPEND "${v}" " ${_LLVMFlang_LINK_DIRS} ${_LLVMFlang_LINK_LIBS} ${_LLVMFlang_LINK_RUNTIME}")
  endforeach()
  unset(_LLVMFlang_LINK_DIRS)
  unset(_LLVMFlang_LINK_LIBS)
  unset(_LLVMFlang_LINK_RUNTIME)
else()
  message(FATAL_ERROR "LLVMFlang target ABI unrecognized: ${CMAKE_Fortran_SIMULATE_ID}")
endif()
