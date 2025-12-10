# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# support for the MS assembler, marmasm and marmasm64

# Load the generic ASMInformation file:
set(ASM_DIALECT "_MARMASM")
include(Internal/CMakeASMLinkerInformation)
set(ASM_DIALECT)
