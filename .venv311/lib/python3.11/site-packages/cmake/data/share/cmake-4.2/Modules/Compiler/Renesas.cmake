
include_guard()

include(Compiler/CMakeCommonCompilerMacros)

macro(__compiler_renesas lang)

  if(CMAKE_${lang}_COMPILER_ARCHITECTURE_ID STREQUAL "RX")
# CC-RX
    # Pass directly include and define flag to the assembler.
    if ("${lang}" STREQUAL "ASM")
      set(_ASM_PREFIX "-asmopt=")
    endif()
    set(CMAKE_INCLUDE_FLAG_${lang} "${_ASM_PREFIX}-include=")
    set(CMAKE_${lang}_DEFINE_FLAG "${_ASM_PREFIX}-define=")

    set(CMAKE_LINK_LIBRARY_FLAG "-lnkopt=-library=")
    set(CMAKE_${lang}_RESPONSE_FILE_FLAG "-subcommand=")
    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-lnkopt=" "")

    set(_RENESAS_DEBUG_FLAG "-debug")
    if (CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL 3.02)
      string(APPEND _RENESAS_DEBUG_FLAG " -g_line")
    endif()

    string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
    string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -optimize=0 ${_RENESAS_DEBUG_FLAG}")
    string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " ${CMAKE_${lang}_DEFINE_FLAG}NDEBUG=1")
    string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " ${_RENESAS_DEBUG_FLAG} ${CMAKE_${lang}_DEFINE_FLAG}NDEBUG=1")
    string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -size ${CMAKE_${lang}_DEFINE_FLAG}NDEBUG=1")

    set(_PREP_OUT "-output=prep=<PREPROCESSED_SOURCE>")
    set(_ASM_OUT "-output=src=<ASSEMBLY_SOURCE>")
    set(_OBJ_OUT "-output=obj=<OBJECT>")
    set(_EXE_OUT "-output=abs=<TARGET>")
    if ("${lang}" STREQUAL "ASM")
      # Assembler dependency.
      set(_DEP_OUT "${_ASM_PREFIX}-MM ${_ASM_PREFIX}-MT=<OBJECT> ${_ASM_PREFIX}-MF=<DEP_FILE>")
    else()
      # Compiler dependency.
      set(_DEP_OUT "-MM -MT=<OBJECT> -output=dep=<DEP_FILE>")
    endif()

  elseif(CMAKE_${lang}_COMPILER_ARCHITECTURE_ID STREQUAL "RL78")
# CC-RL
    # Pass directly include and define flag to the assembler.
    if ("${lang}" STREQUAL "ASM")
      set(_ASM_PREFIX "-asmopt=")
      set(CMAKE_INCLUDE_FLAG_${lang} "${_ASM_PREFIX}-include=")
      set(CMAKE_${lang}_DEFINE_FLAG "${_ASM_PREFIX}-define=")
    endif()

    set(CMAKE_LINK_LIBRARY_FLAG "-lnkopt=-library=")
    set(CMAKE_${lang}_RESPONSE_FILE_FLAG "-subcommand=")
    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-lnkopt=" "")

    set(_RENESAS_DEBUG_FLAG "-g")
    if (CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL 1.02)
      string(APPEND _RENESAS_DEBUG_FLAG " -g_line")
    endif()

    string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
    string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -Onothing ${_RENESAS_DEBUG_FLAG}")
    string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " ${CMAKE_${lang}_DEFINE_FLAG}NDEBUG")
    string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " ${_RENESAS_DEBUG_FLAG} ${CMAKE_${lang}_DEFINE_FLAG}NDEBUG")
    string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -Osize ${CMAKE_${lang}_DEFINE_FLAG}NDEBUG")

    set(_PREP_OUT "-P -o <PREPROCESSED_SOURCE>")
    set(_ASM_OUT "-S -o <ASSEMBLY_SOURCE>")
    set(_OBJ_OUT "-c -o <OBJECT>")
    set(_EXE_OUT "-o <TARGET>")
    set(_DEP_OUT "-M -MT=<OBJECT> -o <DEP_FILE>")
    # Assembler dependency. -c is required to avoid to process to link
    if ("${lang}" STREQUAL "ASM")
      set(_DEP_OUT "-c ${_ASM_PREFIX}-MM ${_ASM_PREFIX}-MT=<OBJECT> ${_ASM_PREFIX}-MF=<DEP_FILE>")
    endif()

  elseif(CMAKE_${lang}_COMPILER_ARCHITECTURE_ID STREQUAL "RH850")
# CC-RH
    set(CMAKE_LINK_LIBRARY_FLAG "-Xlk_option=-library=")
    set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Xlk_option=" "")
    # Pass directly include and define flag to the assembler.
    if ("${lang}" STREQUAL "ASM")
      set(_ASM_PREFIX "-Xasm_option=")
      set(CMAKE_INCLUDE_FLAG_${lang} "${_ASM_PREFIX}-I")
      set(CMAKE_${lang}_DEFINE_FLAG "${_ASM_PREFIX}-D")
    endif()

    set(_RENESAS_DEBUG_FLAG "-g")
    if (CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL 1.05)
      string(APPEND _RENESAS_DEBUG_FLAG " -g_line")
    endif()

    string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
    string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -Onothing ${_RENESAS_DEBUG_FLAG}")
    string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " ${CMAKE_${lang}_DEFINE_FLAG}NDEBUG")
    string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " ${_RENESAS_DEBUG_FLAG} ${CMAKE_${lang}_DEFINE_FLAG}NDEBUG")
    string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -Osize ${CMAKE_${lang}_DEFINE_FLAG}NDEBUG")

    set(_PREP_OUT "-P -o<PREPROCESSED_SOURCE>")
    set(_ASM_OUT "-S -o<ASSEMBLY_SOURCE>")
    set(_OBJ_OUT "-c -o<OBJECT>")
    set(_EXE_OUT "-o<TARGET>")
    set(_DEP_OUT "-M -o<DEP_FILE>")
    # Assembler dependency. -c is required to avoid to process to link
    if ("${lang}" STREQUAL "ASM")
      set(_DEP_OUT "-c ${_ASM_PREFIX}-MM ${_ASM_PREFIX}-MT=<OBJECT> ${_ASM_PREFIX}-MF=<DEP_FILE>")
    endif()

# Otherwise, not supported architecture.
  else()
    message(FATAL_ERROR "Architecture for Renesas compiler: ${CMAKE_${lang}_COMPILER_ARCHITECTURE_ID} is not supported.")
  endif()

# Common
  set(CMAKE_EXECUTABLE_SUFFIX ".abs")
  set(CMAKE_EXECUTABLE_SUFFIX_${lang} ".abs")
  set(CMAKE_STATIC_LIBRARY_PREFIX "")
  set(CMAKE_STATIC_LIBRARY_SUFFIX ".lib")
  set(CMAKE_${lang}_DEPENDS_USE_COMPILER TRUE)
  set(CMAKE_${lang}_DEPFILE_FORMAT custom)
  # This is supported only when link process is called from driver.
  set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "${CMAKE_${lang}_RESPONSE_FILE_FLAG}")

# Compile commands
  set(CMAKE_${lang}_CREATE_PREPROCESSED_SOURCE "<CMAKE_${lang}_COMPILER> <SOURCE> <DEFINES> <INCLUDES> <FLAGS> ${_PREP_OUT}")
  set(CMAKE_${lang}_CREATE_ASSEMBLY_SOURCE "<CMAKE_${lang}_COMPILER> <SOURCE> <DEFINES> <INCLUDES> <FLAGS> ${_ASM_OUT}")
  set(CMAKE_${lang}_COMPILE_OBJECT "<CMAKE_${lang}_COMPILER> <SOURCE> <DEFINES> <INCLUDES> <FLAGS> ${_OBJ_OUT}")
  set(CMAKE_${lang}_LINK_EXECUTABLE "<CMAKE_${lang}_COMPILER> <FLAGS> <LINK_FLAGS> <OBJECTS> <LINK_LIBRARIES> ${_EXE_OUT}")
  set(CMAKE_${lang}_DEPENDS_EXTRA_COMMANDS "<CMAKE_${lang}_COMPILER> <SOURCE> <DEFINES> <INCLUDES> <FLAGS> ${_DEP_OUT}")

# Link/Archive commands. Use rlink as the linker and archiver.
  get_filename_component(_RENESAS_CC_PATH ${CMAKE_${lang}_COMPILER} DIRECTORY NO_CACHE)
  set(_RLINK "\"${_RENESAS_CC_PATH}/rlink\"")
  set(CMAKE_${lang}_CREATE_STATIC_LIBRARY "${_RLINK} -nologo -form=library -output=<TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_CREATE "${_RLINK} -nologo -form=library -output=<TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_APPEND "${_RLINK} -nologo -form=library -output=<TARGET> -library=<TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_FINISH "")

endmacro()
