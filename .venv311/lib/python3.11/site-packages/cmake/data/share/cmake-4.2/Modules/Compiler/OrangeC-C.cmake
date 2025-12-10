include(Compiler/OrangeC)
include(Compiler/CMakeCommonCompilerMacros)

set(CMAKE_C_OUTPUT_EXTENSION ".o")
set(CMAKE_C_VERBOSE_FLAG "-yyyyy")
set(CMAKE_C_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -x c)

set(CMAKE_C90_STANDARD_COMPILE_OPTION -std=c89)
set(CMAKE_C90_EXTENSION_COMPILE_OPTION -std=c89)
set(CMAKE_C90_STANDARD__HAS_FULL_SUPPORT ON)
set(CMAKE_C99_STANDARD_COMPILE_OPTION -std=c99)
set(CMAKE_C99_EXTENSION_COMPILE_OPTION -std=c99)
set(CMAKE_C99_STANDARD__HAS_FULL_SUPPORT ON)
set(CMAKE_C11_STANDARD_COMPILE_OPTION -std=c11)
set(CMAKE_C11_EXTENSION_COMPILE_OPTION -std=c11)
set(CMAKE_C11_STANDARD__HAS_FULL_SUPPORT ON)

set(CMAKE_C_STANDARD_LATEST 11)

__compiler_orangec(C)
#- 6.38 is the earliest version which version info is available in the preprocessor
__compiler_check_default_language_standard(C 6.38 11)
