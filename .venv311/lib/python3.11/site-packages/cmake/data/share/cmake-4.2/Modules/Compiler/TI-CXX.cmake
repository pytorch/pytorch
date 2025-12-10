include(Compiler/TI)
__compiler_ti(CXX)

# Architecture specific

if("${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "ARM")
  set(__COMPILER_TI_CXX03_VERSION 5.2)
  set(__COMPILER_TI_CXX14_VERSION 18.1)

elseif("${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "MSP430")
  set(__COMPILER_TI_CXX03_VERSION 4.4)
  set(__COMPILER_TI_CXX14_VERSION 18.1)

elseif("${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "TMS320C28x")
  set(__COMPILER_TI_CXX03_VERSION 16.9)

elseif("${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "TMS320C6x")
  set(__COMPILER_TI_CXX03_VERSION 8.1)
  set(__COMPILER_TI_CXX14_VERSION 8.3)

else()
  # architecture not handled
  return()

endif()


if(DEFINED __COMPILER_TI_CXX14_VERSION AND
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "${__COMPILER_TI_CXX14_VERSION}")

  # C++03 is not supported anymore
  set(CMAKE_CXX98_STANDARD_COMPILE_OPTION  "--strict_ansi")
  set(CMAKE_CXX98_EXTENSION_COMPILE_OPTION "--relaxed_ansi")

  # C++11 was never supported
  set(CMAKE_CXX11_STANDARD_COMPILE_OPTION "--strict_ansi")
  set(CMAKE_CXX11_EXTENSION_COMPILE_OPTION "--relaxed_ansi")

  set(CMAKE_CXX14_STANDARD_COMPILE_OPTION "--c++14" "--strict_ansi")
  set(CMAKE_CXX14_EXTENSION_COMPILE_OPTION "--c++14" "--relaxed_ansi")

  set(CMAKE_CXX_STANDARD_LATEST 14)

elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "${__COMPILER_TI_CXX03_VERSION}")

  set(CMAKE_CXX98_STANDARD_COMPILE_OPTION "--c++03" "--strict_ansi")
  set(CMAKE_CXX98_EXTENSION_COMPILE_OPTION "--c++03" "--relaxed_ansi")
  set(CMAKE_CXX_STANDARD_LATEST 98)

else()

  set(CMAKE_CXX98_STANDARD_COMPILE_OPTION  "--strict_ansi")
  set(CMAKE_CXX98_EXTENSION_COMPILE_OPTION "--relaxed_ansi")
  set(CMAKE_CXX_STANDARD_LATEST 98)

endif()


# Architecture specific
# CXX98 versions: https://processors.wiki.ti.com/index.php/C%2B%2B_Support_in_TI_Compilers

if("${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "ARM")
  __compiler_check_default_language_standard(CXX 4.5 98 ${__COMPILER_TI_CXX14_VERSION} 14)

elseif("${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "MSP430")
  __compiler_check_default_language_standard(CXX 3.0 98 ${__COMPILER_TI_CXX14_VERSION} 14)

elseif("${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "TMS320C28x")
  __compiler_check_default_language_standard(CXX 5.1 98)

elseif("${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "TMS320C6x")
  __compiler_check_default_language_standard(CXX 6.1 98 ${__COMPILER_TI_CXX14_VERSION} 14)

endif()
