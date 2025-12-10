# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file sets the basic flags for the Windows Resource Compiler.
# It also loads the available platform file for the system-compiler
# if it exists.

# make sure we don't use CMAKE_BASE_NAME from somewhere else
set(CMAKE_BASE_NAME)
if(CMAKE_RC_COMPILER MATCHES "windres[^/]*$")
 set(CMAKE_BASE_NAME "windres")
else()
 get_filename_component(CMAKE_BASE_NAME ${CMAKE_RC_COMPILER} NAME_WE)
endif()
set(CMAKE_SYSTEM_AND_RC_COMPILER_INFO_FILE
  ${CMAKE_ROOT}/Modules/Platform/${CMAKE_SYSTEM_NAME}-${CMAKE_BASE_NAME}.cmake)
include(Platform/${CMAKE_SYSTEM_NAME}-${CMAKE_BASE_NAME} OPTIONAL)

# This should be included before the _INIT variables are
# used to initialize the cache.  Since the rule variables
# have if blocks on them, users can still define them here.
# But, it should still be after the platform file so changes can
# be made to those values.
if(CMAKE_USER_MAKE_RULES_OVERRIDE)
  # Save the full path of the file so try_compile can use it.
  include(${CMAKE_USER_MAKE_RULES_OVERRIDE} RESULT_VARIABLE _override)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE "${_override}")
endif()

set(CMAKE_RC_FLAGS_INIT "$ENV{RCFLAGS} ${CMAKE_RC_FLAGS_INIT}")

cmake_initialize_per_config_variable(CMAKE_RC_FLAGS "Flags for Windows Resource Compiler")

# These are the only types of flags that should be passed to the rc
# command, if COMPILE_FLAGS is used on a target this will be used
# to filter out any other flags
set(CMAKE_RC_FLAG_REGEX "^[-/](D|I)")

# now define the following rule variables
# CMAKE_RC_COMPILE_OBJECT
set(CMAKE_INCLUDE_FLAG_RC "-I ")
# compile a Resource file into an object file
if(NOT CMAKE_RC_COMPILE_OBJECT)
  set(CMAKE_RC_COMPILE_OBJECT
    "<CMAKE_RC_COMPILER> <DEFINES> <INCLUDES> <FLAGS> /fo <OBJECT> <SOURCE>")
endif()

set(CMAKE_RC_USE_LINKER_INFORMATION FALSE)

# set this variable so we can avoid loading this more than once.
set(CMAKE_RC_INFORMATION_LOADED 1)
