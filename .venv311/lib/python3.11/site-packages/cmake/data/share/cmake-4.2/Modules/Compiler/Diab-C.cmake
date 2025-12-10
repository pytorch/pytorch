include(Compiler/Diab)

__compiler_diab(C)

# c89/90 is both -Xdialect-c89
set(CMAKE_C89_STANDARD_COMPILE_OPTION "-Xdialect-c89")
set(CMAKE_C90_STANDARD_COMPILE_OPTION "-Xdialect-c89")
set(CMAKE_C99_STANDARD_COMPILE_OPTION "-Xdialect-c99")
set(CMAKE_C11_STANDARD_COMPILE_OPTION "-Xdialect-c11")
