include(Compiler/OrangeC)
include(Compiler/CMakeCommonCompilerMacros)

set(_ORANGEC_COMPILE_CXX " -x c++")
set(CMAKE_CXX_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -x c++)

set(CMAKE_CXX_OUTPUT_EXTENSION ".o")
set(CMAKE_CXX_VERBOSE_FLAG "-yyyyy")



#- OrangeC is a little lax when accepting compiler version specifications.
#  Usually changing the version only changes the value of __cplusplus.
#  Also we don't support CXX98
set(CMAKE_CXX11_STANDARD_COMPILE_OPTION "-std=c++11")
set(CMAKE_CXX11_EXTENSION_COMPILE_OPTION "-std=c++11")
set(CMAKE_CXX11_STANDARD__HAS_FULL_SUPPORT ON)

set(CMAKE_CXX14_STANDARD_COMPILE_OPTION "-std=c++14")
set(CMAKE_CXX14_EXTENSION_COMPILE_OPTION "-std=c++14")
set(CMAKE_CXX14_STANDARD__HAS_FULL_SUPPORT ON)

set(CMAKE_CXX_STANDARD_LATEST 14)

__compiler_orangec(CXX)
#- 6.38 is the earliest version which version info is available in the preprocessor
__compiler_check_default_language_standard(CXX 6.38 14)
