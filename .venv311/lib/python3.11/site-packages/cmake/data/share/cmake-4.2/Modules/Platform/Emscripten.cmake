# Emscripten provides a combined toolchain file and platform module
# that predates CMake upstream support.  As a toolchain file it sets
# CMAKE_SYSTEM_VERSION to 1 and points CMAKE_MODULE_PATH to itself.
# Include it here to preserve its role as a platform module.
if(CMAKE_SYSTEM_VERSION EQUAL 1 AND CMAKE_MODULE_PATH)
  find_file(_EMSCRIPTEN_PLATFORM_MODULE NAMES "Platform/Emscripten.cmake"
    NO_CACHE NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH PATHS ${CMAKE_MODULE_PATH})
  if(_EMSCRIPTEN_PLATFORM_MODULE)
    include("${_EMSCRIPTEN_PLATFORM_MODULE}")
    unset(_EMSCRIPTEN_PLATFORM_MODULE)
    return()
  endif()
  unset(_EMSCRIPTEN_PLATFORM_MODULE)
endif()

set(CMAKE_SHARED_LIBRARY_LINK_C_WITH_RUNTIME_PATH ON)

set(CMAKE_SHARED_LIBRARY_SUFFIX ".wasm")
set(CMAKE_EXECUTABLE_SUFFIX ".js")

set(CMAKE_DL_LIBS "")
