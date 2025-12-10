# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(Compiler/MSVC)
__compiler_msvc(CXX)

include(Compiler/CMakeCommonCompilerMacros)

if ((CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.0.24215.1 AND
     CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.10) OR
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.10.25017)

  # VS 2015 Update 3 and above support language standard level flags,
  # with the default and minimum level being C++14.
  set(CMAKE_CXX98_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CXX98_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_CXX98_STANDARD__HAS_FULL_SUPPORT ON)
  set(CMAKE_CXX11_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CXX11_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_CXX14_STANDARD_COMPILE_OPTION "-std:c++14")
  set(CMAKE_CXX14_EXTENSION_COMPILE_OPTION "-std:c++14")

  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.11.25505)
    set(CMAKE_CXX11_STANDARD__HAS_FULL_SUPPORT ON)
    set(CMAKE_CXX14_STANDARD__HAS_FULL_SUPPORT ON)
    set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std:c++17")
    set(CMAKE_CXX17_EXTENSION_COMPILE_OPTION "-std:c++17")
  else()
    set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std:c++latest")
    set(CMAKE_CXX17_EXTENSION_COMPILE_OPTION "-std:c++latest")
  endif()

  set(CMAKE_CXX_STANDARD_LATEST 17)

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.29.30129)
    set(CMAKE_CXX20_STANDARD_COMPILE_OPTION "-std:c++20")
    set(CMAKE_CXX20_EXTENSION_COMPILE_OPTION "-std:c++20")
    set(CMAKE_CXX_STANDARD_LATEST 20)
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.12.25835)
    set(CMAKE_CXX20_STANDARD_COMPILE_OPTION "-std:c++latest")
    set(CMAKE_CXX20_EXTENSION_COMPILE_OPTION "-std:c++latest")
    set(CMAKE_CXX_STANDARD_LATEST 20)
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.29.30129)
    set(CMAKE_CXX23_STANDARD_COMPILE_OPTION "-std:c++latest")
    set(CMAKE_CXX23_EXTENSION_COMPILE_OPTION "-std:c++latest")
    set(CMAKE_CXX_STANDARD_LATEST 23)
  endif()

  __compiler_check_default_language_standard(CXX 19.0 14)

elseif (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16.0)
  # MSVC has no specific options to set language standards, but set them as
  # empty strings anyways so the feature test infrastructure can at least check
  # to see if they are defined.
  set(CMAKE_CXX98_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CXX98_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_CXX11_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CXX11_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_CXX14_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CXX14_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CXX17_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_CXX20_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CXX20_EXTENSION_COMPILE_OPTION "")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.0)
    set(CMAKE_CXX_STANDARD_LATEST 17)
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 18.0)
    set(CMAKE_CXX_STANDARD_LATEST 14)
  else()
    set(CMAKE_CXX_STANDARD_LATEST 11)
  endif()

  # There is no meaningful default for this
  set(CMAKE_CXX_STANDARD_DEFAULT "")

  # There are no compiler modes so we only need to test features once.
  # Override the default macro for this special case.  Pretend that
  # all language standards are available so that at least compilation
  # can be attempted.
  macro(cmake_record_cxx_compile_features)
    list(APPEND CMAKE_CXX_COMPILE_FEATURES
      cxx_std_98
      cxx_std_11
      cxx_std_14
      cxx_std_17
      cxx_std_20
      cxx_std_23
      cxx_std_26
      )
    _record_compiler_features(CXX "" CMAKE_CXX_COMPILE_FEATURES)
  endmacro()
endif()

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.34")
  string(CONCAT CMAKE_CXX_SCANDEP_SOURCE
    "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> <SOURCE> -nologo -TP"
    " -showIncludes"
    " -scanDependencies <DYNDEP_FILE>"
    " -Fo<OBJECT>")
  set(CMAKE_CXX_SCANDEP_DEPFILE_FORMAT "msvc")
  set(CMAKE_CXX_MODULE_MAP_FORMAT "msvc")
  set(CMAKE_CXX_MODULE_MAP_FLAG "@<MODULE_MAP_FILE>")
  set(CMAKE_CXX_MODULE_BMI_ONLY_FLAG "-ifcOnly;-ifcOutput;<OBJECT>")
endif ()
