include(Compiler/CMakeCommonCompilerMacros)
include(Compiler/ADSP)

__compiler_adsp(C)

set(CMAKE_C90_STANDARD_COMPILE_OPTION -c89)
set(CMAKE_C90_STANDARD__HAS_FULL_SUPPORT ON)

set(CMAKE_C99_STANDARD__HAS_FULL_SUPPORT ON)

set(CMAKE_C_STANDARD_LATEST 99)

__compiler_check_default_language_standard(C 8.0.0.0 99)
