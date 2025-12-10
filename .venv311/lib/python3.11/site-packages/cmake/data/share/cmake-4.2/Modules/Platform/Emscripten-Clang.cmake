# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include_guard()

macro(__emscripten_clang lang)
  set(CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG "-Wl,-soname,")

  # FIXME(#27240): We do not add -sMAIN_MODULE to CMAKE_${lang}_LINK_EXECUTABLE
  # because it is not always needed, and can break things if added unnecessarily.
  # We also do not add -sMAIN_MODULE to CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS
  # to preserve legacy behavior in which projects added it as needed.
  # In the future we may add both flags with suitable controls.
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "")

  set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_LIBRARIES 1)
  set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_OBJECTS 1)
  set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_INCLUDES 1)
  set(CMAKE_${lang}_COMPILE_OBJECT
    "<CMAKE_${lang}_COMPILER> -c <SOURCE> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -fPIC")
endmacro()
