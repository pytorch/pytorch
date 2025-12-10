# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that the selected ASM_MASM "compiler" (should be masm or masm64)
# works. For assembler this can only check whether the compiler has been found,
# because otherwise there would have to be a separate assembler source file
# for each assembler on every architecture.

set(ASM_DIALECT "_MASM")
include(CMakeTestASMCompiler)
set(ASM_DIALECT)
