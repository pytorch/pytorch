# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file is included by cmGlobalGenerator::EnableLanguage.
# It is included before the compiler has been determined.

# before cmake 2.6 these variables were set in cmMakefile.cxx. This is still
# done to keep scripts and custom language and compiler modules working.
# But they are reset here and set again in the platform files for the target
# platform, so they can be used for testing the target platform instead
# of testing the host platform.
unset(APPLE)
unset(UNIX)
unset(CYGWIN)
unset(MSYS)
unset(WIN32)
unset(BSD)
unset(LINUX)
unset(AIX)

# The CMAKE_EFFECTIVE_SYSTEM_NAME is used to load compiler and compiler
# wrapper configuration files. By default it equals to CMAKE_SYSTEM_NAME
# but could be overridden in the ${CMAKE_SYSTEM_NAME}-Initialize files.
#
# It is useful to share the same aforementioned configuration files and
# avoids duplicating them in case of tightly related platforms.
#
# An example are the platforms supported by Xcode (macOS, iOS, tvOS, visionOS
# and watchOS). For all of those the CMAKE_EFFECTIVE_SYSTEM_NAME is
# set to Apple which results in using
# Platform/Apple-AppleClang-CXX.cmake for the Apple C++ compiler.
set(CMAKE_EFFECTIVE_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")

include(Platform/${CMAKE_SYSTEM_NAME}-Initialize OPTIONAL)

set(CMAKE_SYSTEM_SPECIFIC_INITIALIZE_LOADED 1)
