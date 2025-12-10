# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


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

if(CMAKE_USER_MAKE_RULES_OVERRIDE_Java)
  # Save the full path of the file so try_compile can use it.
  include(${CMAKE_USER_MAKE_RULES_OVERRIDE_Java} RESULT_VARIABLE _override)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE_Java "${_override}")
endif()

# this is a place holder if java needed flags for javac they would go here.
if(NOT CMAKE_Java_CREATE_STATIC_LIBRARY)
#  if(WIN32)
#    set(class_files_mask "*.class")
#  else()
    set(class_files_mask ".")
#  endif()

  set(CMAKE_Java_CREATE_STATIC_LIBRARY
      "<CMAKE_Java_ARCHIVE> -cf <TARGET> -C <OBJECT_DIR> ${class_files_mask}")
    # "${class_files_mask}" should really be "<OBJECTS>" but compiling a *.java
    # file can create more than one *.class file...
endif()

# compile a Java file into an object file
if(NOT CMAKE_Java_COMPILE_OBJECT)
  set(CMAKE_Java_COMPILE_OBJECT
    "<CMAKE_Java_COMPILER> <FLAGS> <SOURCE> -d <OBJECT_DIR>")
endif()

# set java include flag option and the separator for multiple include paths
set(CMAKE_INCLUDE_FLAG_Java "-classpath ")
if(WIN32 AND NOT CYGWIN)
  set(CMAKE_INCLUDE_FLAG_SEP_Java ";")
else()
  set(CMAKE_INCLUDE_FLAG_SEP_Java ":")
endif()

set(CMAKE_Java_USE_LINKER_INFORMATION FALSE)
