# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# support for AT&T syntax assemblers, e.g. GNU as

set(ASM_DIALECT "-ATT")
# *.S files are supposed to be preprocessed, so they should not be passed to
# assembler but should be processed by gcc
set(CMAKE_ASM${ASM_DIALECT}_SOURCE_FILE_EXTENSIONS s;asm)

set(CMAKE_ASM${ASM_DIALECT}_COMPILE_OBJECT "<CMAKE_ASM${ASM_DIALECT}_COMPILER> <INCLUDES> <FLAGS> -o <OBJECT> <SOURCE>")

include(CMakeASMInformation)
set(ASM_DIALECT)
