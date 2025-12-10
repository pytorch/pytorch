# To include compiler feature detection
include(Compiler/GNU-CXX)

include(Compiler/QCC)
__compiler_qcc(CXX)

# If the toolchain uses qcc for CMAKE_CXX_COMPILER instead of QCC, the
# default for the driver is not c++.
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.2.0") # QNX 8.0 toolchain
  set(_cmake_qcc_cxx_lang_compile_flag "-lang-c++")
  set(_cmake_qcc_cxx_lang_link_flag "-lang-c++")
else ()
  set(_cmake_qcc_cxx_lang_compile_flag "-x c++")
  set(_cmake_qcc_cxx_lang_link_flag "")
endif ()
set(CMAKE_CXX_COMPILE_OBJECT
  "<CMAKE_CXX_COMPILER> ${_cmake_qcc_cxx_lang_compile_flag} <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
set(CMAKE_CXX_LINK_EXECUTABLE
  "<CMAKE_CXX_COMPILER> ${_cmake_qcc_cxx_lang_link_flag} <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
unset(_cmake_qcc_cxx_lang_compile_flag)
unset(_cmake_qcc_cxx_lang_link_flag)

set(CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-fvisibility-inlines-hidden")
