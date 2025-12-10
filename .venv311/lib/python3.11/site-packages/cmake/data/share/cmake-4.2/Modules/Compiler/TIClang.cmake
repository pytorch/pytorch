if(__COMPILER_TICLANG)
  return()
endif()
set(__COMPILER_TICLANG TRUE)

include(Compiler/CMakeCommonCompilerMacros)

# get linker supported cpu list
macro(__compiler_ticlang lang)
  set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "-Xlinker ")

  set(CMAKE_${lang}_CREATE_PREPROCESSED_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E <SOURCE> > <PREPROCESSED_SOURCE>")

  set(CMAKE_${lang}_CREATE_ASSEMBLY_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -S <SOURCE> -o <ASSEMBLY_SOURCE>")

  set(CMAKE_${lang}_COMPILE_OBJECT  "<CMAKE_${lang}_COMPILER> -c <SOURCE> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT>")

  set(CMAKE_${lang}_LINK_EXECUTABLE "<CMAKE_${lang}_COMPILER> <FLAGS> -Xlinker --output_file=<TARGET> -Xlinker --map_file=<TARGET_NAME>.map -Xlinker --rom_model <LINK_FLAGS> <OBJECTS> <LINK_LIBRARIES>")

  set(CMAKE_${lang}_ARCHIVE_CREATE  "<CMAKE_AR> cr <TARGET> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_APPEND  "<CMAKE_AR> r <TARGET> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_FINISH  "")

  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Xlinker" " ")
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP)

  set(CMAKE_${lang}_LINK_MODE DRIVER)
endmacro()

set(CMAKE_EXECUTABLE_SUFFIX ".out")
set(CMAKE_LIBRARY_PATH_FLAG "-Wl,--search_path=")
set(CMAKE_LINK_LIBRARY_FLAG "-Wl,--library=")
