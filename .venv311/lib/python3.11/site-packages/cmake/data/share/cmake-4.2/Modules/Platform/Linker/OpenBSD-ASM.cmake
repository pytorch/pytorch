# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(Platform/Linker/BSD-Linker-Initialize)

if(_CMAKE_SYSTEM_LINKER_TYPE STREQUAL "GNU")
  include(Platform/Linker/OpenBSD-GNU-ASM)
else()
  include(Platform/Linker/OpenBSD-LLD-ASM)
endif()
