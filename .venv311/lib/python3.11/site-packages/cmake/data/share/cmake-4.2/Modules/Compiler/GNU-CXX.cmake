include(Compiler/GNU)
__compiler_gnu(CXX)
__compiler_gnu_cxx_standards(CXX)


if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
    AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
    AND CMAKE_DEPFILE_FLAGS_CXX)
  # dependencies are computed by the compiler itself
  set(CMAKE_CXX_DEPFILE_FORMAT gcc)
  set(CMAKE_CXX_DEPENDS_USE_COMPILER TRUE)
endif()

set(CMAKE_CXX_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -x c++)

if (WIN32)
  if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.6)
    set(CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-fno-keep-inline-dllexport")
  endif()
else()
  if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.0)
    set(CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-fvisibility-inlines-hidden")
  endif()
endif()

__compiler_check_default_language_standard(CXX 3.4 98 6.0 14 11.1 17)

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0)
  string(CONCAT CMAKE_CXX_SCANDEP_SOURCE
    "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E -x c++ <SOURCE>"
    " -MT <DYNDEP_FILE> -MD -MF <DEP_FILE>"
    " -fmodules-ts -fdeps-file=<DYNDEP_FILE> -fdeps-target=<OBJECT> -fdeps-format=p1689r5"
    " -o <PREPROCESSED_SOURCE>")
  set(CMAKE_CXX_MODULE_MAP_FORMAT "gcc")
  string(CONCAT CMAKE_CXX_MODULE_MAP_FLAG
    # Turn on modules.
    "-fmodules-ts"
    # Read the module mapper file.
    " -fmodule-mapper=<MODULE_MAP_FILE>"
    # Make sure dependency tracking is enabled (missing from `try_*`).
    " -MD"
    # Suppress `CXX_MODULES +=` from generated depfile snippets.
    " -fdeps-format=p1689r5"
    # Force C++ as a language.
    " -x c++")
  set(CMAKE_CXX_MODULE_BMI_ONLY_FLAG "-fmodule-only")
endif()
