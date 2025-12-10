# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# Do not include this module directly from code outside CMake!
set(_JAVA_HOME "")
if(JAVA_HOME AND IS_DIRECTORY "${JAVA_HOME}")
  set(_JAVA_HOME "${JAVA_HOME}")
  set(_JAVA_HOME_EXPLICIT 1)
else()
  set(_ENV_JAVA_HOME "")
  if(DEFINED ENV{JAVA_HOME})
    file(TO_CMAKE_PATH "$ENV{JAVA_HOME}" _ENV_JAVA_HOME)
  endif()
  if(_ENV_JAVA_HOME AND IS_DIRECTORY "${_ENV_JAVA_HOME}")
    set(_JAVA_HOME "${_ENV_JAVA_HOME}")
    set(_JAVA_HOME_EXPLICIT 1)
  else()
    set(_CMD_JAVA_HOME "")
    if(APPLE AND EXISTS /usr/libexec/java_home)
      execute_process(COMMAND /usr/libexec/java_home
        OUTPUT_VARIABLE _CMD_JAVA_HOME
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
    endif()
    if(_CMD_JAVA_HOME AND IS_DIRECTORY "${_CMD_JAVA_HOME}")
      set(_JAVA_HOME "${_CMD_JAVA_HOME}")
      set(_JAVA_HOME_EXPLICIT 0)
    endif()
    unset(_CMD_JAVA_HOME)
  endif()
  unset(_ENV_JAVA_HOME)
endif()
