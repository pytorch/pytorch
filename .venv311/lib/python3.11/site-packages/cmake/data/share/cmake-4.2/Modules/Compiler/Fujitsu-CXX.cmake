# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.
include(Compiler/Fujitsu)
__compiler_fujitsu(CXX)

#set(CMAKE_PCH_EXTENSION .pch)
#set(CMAKE_PCH_EPILOGUE "#pragma hdrstop")
#set(CMAKE_CXX_COMPILE_OPTIONS_USE_PCH --no_pch_messages -include <PCH_HEADER> --use_pch <PCH_FILE>)
#set(CMAKE_CXX_COMPILE_OPTIONS_CREATE_PCH --no_pch_messages -include <PCH_HEADER> --create_pch <PCH_FILE>)

# The Fujitsu compiler offers both a 98 and 03 mode.  These two are
# essentially interchangeable as 03 simply provides clarity to some 98
# ambiguyity.
#
# Re: Stroustrup's C++ FAQ:
#   What is the difference between C++98 and C++03?
#     From a programmer's view there is none. The C++03 revision of the
#     standard was a bug fix release for implementers to ensure greater
#     consistency and portability. In particular, tutorial and reference
#     material describing C++98 and C++03 can be used interchangeably by all
#     except compiler writers and standards gurus.
#
# Since CMake doesn't actually have an 03 mode and they're effectively
# interchangeable then we're just going to explicitly use 03 mode in the
# compiler when 98 is requested.

# The version matching is messy here.  The std support seems to be related to
# the compiler tweak version derived from the patch id in the version string.

if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 4)
  set(CMAKE_CXX98_STANDARD_COMPILE_OPTION  -std=c++03)
  set(CMAKE_CXX98_EXTENSION_COMPILE_OPTION -std=gnu++03)
  set(CMAKE_CXX98_STANDARD__HAS_FULL_SUPPORT ON)

  set(CMAKE_CXX11_STANDARD_COMPILE_OPTION  -std=c++11)
  set(CMAKE_CXX11_EXTENSION_COMPILE_OPTION -std=gnu++11)
  set(CMAKE_CXX11_STANDARD__HAS_FULL_SUPPORT ON)

  set(CMAKE_CXX14_STANDARD_COMPILE_OPTION  -std=c++14)
  set(CMAKE_CXX14_EXTENSION_COMPILE_OPTION -std=gnu++14)
  set(CMAKE_CXX14_STANDARD__HAS_FULL_SUPPORT ON)

  set(CMAKE_CXX17_STANDARD_COMPILE_OPTION  -std=c++17)
  set(CMAKE_CXX17_EXTENSION_COMPILE_OPTION -std=gnu++17)

  set(CMAKE_CXX_STANDARD_LATEST 17)
endif()

__compiler_check_default_language_standard(CXX 4 14)
