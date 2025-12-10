# IAR C/C++ Compiler (https://www.iar.com)
# CPU <arch> supported in CMake: 8051, Arm, AVR, MSP430, RH850, RISC-V, RL78, RX and V850
#
# IAR C/C++ Compiler for <arch> internal integer symbols used in CMake:
#
# __IAR_SYSTEMS_ICC__
#           Provides the compiler internal platform version
# __ICC<arch>__
#           Provides 1 for the current <arch> in use
# __VER__
#           Provides the current version in use
#            The semantic version of the compiler is architecture-dependent
#            When <arch> is ARM:
#               CMAKE_<LANG>_COMPILER_VERSION_MAJOR = (__VER__ / 1E6)
#               CMAKE_<LANG>_COMPILER_VERSION_MINOR = (__VER__ / 1E3) % 1E3
#               CMAKE_<LANG>_COMPILER_VERSION_PATCH = (__VER__ % 1E3)
#            When <arch> is non-ARM:
#               CMAKE_<LANG>_COMPILER_VERSION_MAJOR = (__VER__ / 1E2)
#               CMAKE_<LANG>_COMPILER_VERSION_MINOR = (__VER__ - ((__VER__/ 1E2) * 1E2))
#               CMAKE_<LANG>_COMPILER_VERSION_PATCH = (__SUBVERSION__)
# __SUBVERSION__
#           Provides the version's patch level for non-ARM <arch>
#
set(_compiler_id_pp_test "defined(__IAR_SYSTEMS_ICC__) || defined(__IAR_SYSTEMS_ICC)")

set(_compiler_id_version_compute "
# if defined(__VER__) && defined(__ICCARM__)
#  define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@((__VER__) / 1000000)
#  define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(((__VER__) / 1000) % 1000)
#  define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@((__VER__) % 1000)
#  define @PREFIX@COMPILER_VERSION_INTERNAL @MACRO_DEC@(__IAR_SYSTEMS_ICC__)
# elif defined(__VER__) && (defined(__ICCAVR__) || defined(__ICCRX__) || defined(__ICCRH850__) || defined(__ICCRL78__) || defined(__ICC430__) || defined(__ICCRISCV__) || defined(__ICCV850__) || defined(__ICC8051__) || defined(__ICCSTM8__))
#  define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@((__VER__) / 100)
#  define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@((__VER__) - (((__VER__) / 100)*100))
#  define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__SUBVERSION__)
#  define @PREFIX@COMPILER_VERSION_INTERNAL @MACRO_DEC@(__IAR_SYSTEMS_ICC__)
# endif")
