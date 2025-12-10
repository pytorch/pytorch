if(_ARMCC_CMAKE_LOADED)
  return()
endif()
set(_ARMCC_CMAKE_LOADED TRUE)

# See ARM Compiler documentation at:
# http://infocenter.arm.com/help/topic/com.arm.doc.set.swdev/index.html

get_filename_component(_CMAKE_C_TOOLCHAIN_LOCATION "${CMAKE_C_COMPILER}" PATH)
get_filename_component(_CMAKE_CXX_TOOLCHAIN_LOCATION "${CMAKE_CXX_COMPILER}" PATH)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

find_program(CMAKE_ARMCC_LINKER armlink HINTS "${_CMAKE_C_TOOLCHAIN_LOCATION}" "${_CMAKE_CXX_TOOLCHAIN_LOCATION}" )
find_program(CMAKE_ARMCC_AR     armar   HINTS "${_CMAKE_C_TOOLCHAIN_LOCATION}" "${_CMAKE_CXX_TOOLCHAIN_LOCATION}" )

set(CMAKE_LINKER "${CMAKE_ARMCC_LINKER}" CACHE FILEPATH "The ARMCC linker" FORCE)
mark_as_advanced(CMAKE_ARMCC_LINKER)
set(CMAKE_AR "${CMAKE_ARMCC_AR}" CACHE FILEPATH "The ARMCC archiver" FORCE)
mark_as_advanced(CMAKE_ARMCC_AR)

macro(__compiler_armcc lang)
  string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
  string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -g")
  string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -Ospace -DNDEBUG")
  string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -Otime -DNDEBUG")
  string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -O2 -g")

  set(CMAKE_${lang}_OUTPUT_EXTENSION ".o")
  set(CMAKE_${lang}_OUTPUT_EXTENSION_REPLACE 1)
  set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "--via=")

  set(CMAKE_${lang}_LINK_MODE LINKER)

  set(CMAKE_${lang}_LINK_EXECUTABLE      "<CMAKE_LINKER> <LINK_FLAGS> <LINK_LIBRARIES> <OBJECTS> -o <TARGET> --list <TARGET_BASE>.map")
  set(CMAKE_${lang}_CREATE_STATIC_LIBRARY  "<CMAKE_AR> --create -cr <TARGET> <LINK_FLAGS> <OBJECTS>")

  set(CMAKE_DEPFILE_FLAGS_${lang} "--depend=<DEP_FILE> --depend_single_line --no_depend_system_headers")

  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Xlinker" " ")
endmacro()
