include(Compiler/Diab)

__compiler_diab(CXX)

# Diab C++98 is named as -Xdialect-c++03
set(CMAKE_CXX98_STANDARD_COMPILE_OPTION "-Xdialect-c++03")
set(CMAKE_CXX11_STANDARD_COMPILE_OPTION "-Xdialect-c++11")
set(CMAKE_CXX14_STANDARD_COMPILE_OPTION "-Xdialect-c++14")
set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-Xdialect-c++17")
set(CMAKE_CXX20_STANDARD_COMPILE_OPTION "-Xdialect-c++20")

__compiler_check_default_language_standard(CXX 4.0 98 5.0 11)
