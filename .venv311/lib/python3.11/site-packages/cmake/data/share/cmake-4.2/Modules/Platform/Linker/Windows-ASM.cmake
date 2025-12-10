# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

if(CMAKE_ASM_SIMULATE_ID STREQUAL "MSVC")
  # MSVC is the default linker
  include(Platform/Linker/Windows-MSVC-ASM)
else()
    # GNU is the default linker
  include(Platform/Linker/Windows-GNU-ASM)
endif()
