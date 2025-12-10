
include(Compiler/Renesas)
__compiler_renesas(C)

set(CMAKE_C90_STANDARD_COMPILE_OPTION "-lang=c")
set(CMAKE_C90_EXTENSION_COMPILE_OPTION "-lang=c")
set(CMAKE_C99_STANDARD_COMPILE_OPTION "-lang=c99")
set(CMAKE_C99_EXTENSION_COMPILE_OPTION "-lang=c99")

__compiler_check_default_language_standard(C 1.0 90)
