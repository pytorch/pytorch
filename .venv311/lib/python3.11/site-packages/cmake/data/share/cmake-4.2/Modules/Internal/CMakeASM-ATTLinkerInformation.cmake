# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# support for AT&T syntax assemblers, e.g. GNU as

# Load the generic ASMInformation file:
set(ASM_DIALECT "-ATT")
include(Internal/CMakeASMLinkerInformation)
set(ASM_DIALECT)
