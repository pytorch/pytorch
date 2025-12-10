include(Compiler/PGI-C)
include(Compiler/NVHPC)

# Needed so that we support `LANGUAGE` property correctly
set(CMAKE_C_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -x c)
set(CMAKE_C_COMPILE_OPTIONS_VISIBILITY "-fvisibility=")

if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 20.11)
  set(CMAKE_C17_STANDARD_COMPILE_OPTION  -std=c17)
  set(CMAKE_C17_EXTENSION_COMPILE_OPTION -std=gnu17)
  set(CMAKE_C_STANDARD_LATEST 17)
endif()

if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 21.07)
  set(CMAKE_DEPFILE_FLAGS_C "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
  set(CMAKE_C_DEPFILE_FORMAT gcc)
  set(CMAKE_C_DEPENDS_USE_COMPILER TRUE)
else()
  # Before NVHPC 21.07 the `-MD` flag implicitly
  # implies `-E` and therefore compilation and dependency generation
  # can't occur in the same invocation
  set(CMAKE_C_DEPENDS_EXTRA_COMMANDS "<CMAKE_C_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -x c -M <SOURCE> -MT <OBJECT> -MD<DEP_FILE>")
endif()

__compiler_nvhpc(C)
