# Help CMAKE_PARSE_IMPLICIT_LINK_INFO detect NAG Fortran object files.
if(NOT CMAKE_Fortran_COMPILER_WORKS AND NOT CMAKE_Fortran_COMPILER_FORCED)
  message(CHECK_START "Detecting NAG Fortran directory")
  # Run with -dryrun to see sample "link" line.
  execute_process(
    COMMAND ${CMAKE_Fortran_COMPILER} dummy.o -dryrun
    OUTPUT_VARIABLE _dryrun
    ERROR_VARIABLE _dryrun
    )
  # Match an object file.
  string(REGEX MATCH "/[^ ]*/[^ /][^ /]*\\.o" _nag_obj "${_dryrun}")
  if(_nag_obj)
    # Parse object directory and convert to a regex.
    string(REGEX REPLACE "/[^/]*$" "" _nag_dir "${_nag_obj}")
    string(REGEX REPLACE "([][+.*()^])" "\\\\\\1" _nag_regex "${_nag_dir}")
    set(CMAKE_Fortran_IMPLICIT_OBJECT_REGEX "^${_nag_regex}/")
    message(CONFIGURE_LOG
      "Detecting NAG Fortran directory with -dryrun found\n"
      "  object: ${_nag_obj}\n"
      "  directory: ${_nag_dir}\n"
      "  regex: ${CMAKE_Fortran_IMPLICIT_OBJECT_REGEX}\n"
      "from output:\n${_dryrun}\n\n")
    message(CHECK_PASS "${_nag_dir}")
  else()
    message(CONFIGURE_LOG
      "Detecting NAG Fortran directory with -dryrun failed:\n${_dryrun}\n\n")
    message(CHECK_FAIL "failed")
  endif()
endif()

# Initial configuration flags.
string(APPEND CMAKE_Fortran_FLAGS_INIT " ")
string(APPEND CMAKE_Fortran_FLAGS_DEBUG_INIT " -g")
string(APPEND CMAKE_Fortran_FLAGS_MINSIZEREL_INIT " -O2 -DNDEBUG")
string(APPEND CMAKE_Fortran_FLAGS_RELEASE_INIT " -O4 -DNDEBUG")
string(APPEND CMAKE_Fortran_FLAGS_RELWITHDEBINFO_INIT " -O2 -g -DNDEBUG")

set(CMAKE_Fortran_SUBMODULE_SEP ".")
set(CMAKE_Fortran_SUBMODULE_EXT ".sub")
set(CMAKE_Fortran_MODDIR_FLAG "-mdir ")
set(CMAKE_Fortran_MODDIR_INCLUDE_FLAG "-I") # -mdir does not affect search path
set(CMAKE_SHARED_LIBRARY_Fortran_FLAGS "-PIC")
set(CMAKE_Fortran_FORMAT_FIXED_FLAG "-fixed")
set(CMAKE_Fortran_FORMAT_FREE_FLAG "-free")
set(CMAKE_Fortran_COMPILE_OPTIONS_PIC "-PIC")
set(CMAKE_Fortran_COMPILE_OPTIONS_PIE "-PIC")
set(CMAKE_Fortran_RESPONSE_FILE_LINK_FLAG "-Wl,@")
set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_ON "-fpp")

set(CMAKE_Fortran_LINK_MODE DRIVER)

set(CMAKE_Fortran_PREPROCESS_SOURCE
  "<CMAKE_Fortran_COMPILER> -fpp <DEFINES> <INCLUDES> <FLAGS> -F <SOURCE> -o <PREPROCESSED_SOURCE>")
