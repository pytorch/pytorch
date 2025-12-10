# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# Function to compile a source file to identify the compiler ABI.
# This is used internally by CMake and should not be included by user
# code.

include(${CMAKE_ROOT}/Modules/Internal/CMakeDetermineLinkerId.cmake)
include(${CMAKE_ROOT}/Modules/CMakeParseImplicitIncludeInfo.cmake)
include(${CMAKE_ROOT}/Modules/CMakeParseImplicitLinkInfo.cmake)
include(${CMAKE_ROOT}/Modules/CMakeParseLibraryArchitecture.cmake)
include(CMakeTestCompilerCommon)

function(CMAKE_DETERMINE_COMPILER_ABI lang src)
  if(NOT DEFINED CMAKE_${lang}_ABI_COMPILED)
    message(CHECK_START "Detecting ${lang} compiler ABI info")

    # Compile the ABI identification source.
    set(BIN "${CMAKE_PLATFORM_INFO_DIR}/CMakeDetermineCompilerABI_${lang}.bin")
    set(CMAKE_FLAGS )
    set(COMPILE_DEFINITIONS )
    set(LINK_OPTIONS )
    if(DEFINED CMAKE_${lang}_VERBOSE_FLAG)
      set(LINK_OPTIONS "${CMAKE_${lang}_VERBOSE_FLAG}")
      set(COMPILE_DEFINITIONS "${CMAKE_${lang}_VERBOSE_FLAG}")
    endif()
    if(DEFINED CMAKE_${lang}_VERBOSE_COMPILE_FLAG)
      set(COMPILE_DEFINITIONS "${CMAKE_${lang}_VERBOSE_COMPILE_FLAG}")
    endif()
    if(DEFINED CMAKE_${lang}_VERBOSE_LINK_FLAG)
      list(APPEND LINK_OPTIONS "${CMAKE_${lang}_VERBOSE_LINK_FLAG}")
    endif()
    if(lang MATCHES "^(CUDA|HIP)$")
      if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
        # We are about to detect the native architectures, so we do
        # not yet know them.  Use all architectures during detection.
        set(CMAKE_${lang}_ARCHITECTURES "all")
      endif()
      set(CMAKE_${lang}_RUNTIME_LIBRARY "Static")
    endif()
    if(lang STREQUAL "CXX")
      set(CMAKE_${lang}_SCAN_FOR_MODULES OFF)
    endif()
    if(NOT "x${CMAKE_${lang}_COMPILER_ID}" STREQUAL "xMSVC")
      # Avoid adding our own platform standard libraries for compilers
      # from which we might detect implicit link libraries.
      list(APPEND CMAKE_FLAGS "-DCMAKE_${lang}_STANDARD_LIBRARIES=")
    endif()
    list(JOIN LINK_OPTIONS " " LINK_OPTIONS)
    list(APPEND CMAKE_FLAGS "-DEXE_LINKER_FLAGS=${LINK_OPTIONS}")

    __TestCompiler_setTryCompileTargetType()

    # Avoid failing ABI detection caused by non-functionally relevant
    # compiler arguments
    if(CMAKE_TRY_COMPILE_CONFIGURATION)
      string(TOUPPER "${CMAKE_TRY_COMPILE_CONFIGURATION}" _tc_config)
    else()
      set(_tc_config "DEBUG")
    endif()
    foreach(v CMAKE_${lang}_FLAGS CMAKE_${lang}_FLAGS_${_tc_config})
      # Avoid failing ABI detection on warnings.
      string(REGEX REPLACE "(^| )-Werror([= ][^-][^ ]*)?( |$)" " " ${v} "${${v}}")
      # Avoid passing of "-pipe" when determining the compiler internals. With
      # "-pipe" GCC will use pipes to pass data between the involved
      # executables.  This may lead to issues when their stderr output (which
      # contains the relevant compiler internals) becomes interweaved.
      string(REGEX REPLACE "(^| )-pipe( |$)" " " ${v} "${${v}}")
      # Suppress any formatting of warnings and/or errors
      string(REGEX REPLACE "(-f|/)diagnostics(-|:)color(=[a-z]+)?" "" ${v} "${${v}}")
    endforeach()

    # Save the current LC_ALL, LC_MESSAGES, and LANG environment variables
    # and set them to "C" that way GCC's "search starts here" text is in
    # English and we can grok it.
    set(_orig_lc_all      $ENV{LC_ALL})
    set(_orig_lc_messages $ENV{LC_MESSAGES})
    set(_orig_lang        $ENV{LANG})
    set(ENV{LC_ALL}      C)
    set(ENV{LC_MESSAGES} C)
    set(ENV{LANG}        C)
    try_compile(CMAKE_${lang}_ABI_COMPILED
      SOURCES ${src}
      CMAKE_FLAGS ${CMAKE_FLAGS}
                  # Ignore unused flags when we are just determining the ABI.
                  "--no-warn-unused-cli"
      COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS}
      OUTPUT_VARIABLE OUTPUT
      COPY_FILE "${BIN}"
      COPY_FILE_ERROR _copy_error
      __CMAKE_INTERNAL ABI
      )

    # Restore original LC_ALL, LC_MESSAGES, and LANG
    set(ENV{LC_ALL}      ${_orig_lc_all})
    set(ENV{LC_MESSAGES} ${_orig_lc_messages})
    set(ENV{LANG}        ${_orig_lang})

    # Move result from cache to normal variable.
    set(CMAKE_${lang}_ABI_COMPILED ${CMAKE_${lang}_ABI_COMPILED})
    unset(CMAKE_${lang}_ABI_COMPILED CACHE)
    if(CMAKE_${lang}_ABI_COMPILED AND _copy_error)
      set(CMAKE_${lang}_ABI_COMPILED 0)
    endif()
    set(CMAKE_${lang}_ABI_COMPILED ${CMAKE_${lang}_ABI_COMPILED} PARENT_SCOPE)

    # Load the resulting information strings.
    if(CMAKE_${lang}_ABI_COMPILED)
      message(CHECK_PASS "done")
      if(CMAKE_HOST_APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "\\$")
        file(READ_MACHO "${BIN}" ARCHITECTURES archs CAPTURE_ERROR macho_error) # undocumented file() subcommand
        if (NOT macho_error)
          # sort and prune the list of found architectures
          set(arch_list_sorted ${archs})
          list(SORT arch_list_sorted)
          list(REMOVE_DUPLICATES arch_list_sorted)
          # sort and prune the list of requested architectures
          set(requested_arch_list ${CMAKE_OSX_ARCHITECTURES})
          list(SORT requested_arch_list)
          list(REMOVE_DUPLICATES requested_arch_list)
          message(CONFIGURE_LOG
            "Effective list of requested architectures (possibly empty)  : \"${requested_arch_list}\"\n"
            "Effective list of architectures found in the ABI info binary: \"${arch_list_sorted}\"\n")
          # If all generated architectures were known to READ_MACHO (i.e. libmacho):
          # Compare requested and found:
          # - if no architecture(s) were requested explicitly, just check if READ_MACHO returned
          #   an architecture for the ABI info binary.
          # - otherwise, check if the requested and found lists are equal
          if(arch_list_sorted MATCHES "unknown")
            # Rare but not impossible: a host with a toolchain capable of generating binaries with
            # architectures that the system libmacho is too old to know. Report the found archs as
            # usual, warn about the unknowns and skip the comparison with CMAKE_OSX_ARCHITECTURES.
            message(WARNING "The ${lang} compiler generates universal binaries with at least 1 architecture not known to the host")
          elseif(requested_arch_list AND arch_list_sorted
              AND NOT "${requested_arch_list}" STREQUAL "${arch_list_sorted}")
            # inform the user of the mismatch but show the raw input and output lists
            message(FATAL_ERROR
              "The ${lang} compiler targets architectures:\n"
              "  \"${archs}\"\n"
              "but CMAKE_OSX_ARCHITECTURES is\n"
              "  \"${CMAKE_OSX_ARCHITECTURES}\"\n")
          endif()
        endif()
      endif()
      cmake_policy(PUSH)
      cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
      file(STRINGS "${BIN}" ABI_STRINGS LIMIT_COUNT 32 REGEX "INFO:[A-Za-z0-9_]+\\[[^]]*\\]")
      cmake_policy(POP)
      set(ABI_SIZEOF_DPTR "NOTFOUND")
      set(ABI_BYTE_ORDER "NOTFOUND")
      set(ABI_NAME "NOTFOUND")
      set(ARCHITECTURE_ID "")
      foreach(info ${ABI_STRINGS})
        if("${info}" MATCHES "INFO:sizeof_dptr\\[0*([^]]*)\\]" AND NOT ABI_SIZEOF_DPTR)
          set(ABI_SIZEOF_DPTR "${CMAKE_MATCH_1}")
        endif()
        if("${info}" MATCHES "INFO:byte_order\\[(BIG_ENDIAN|LITTLE_ENDIAN)\\]")
          set(byte_order "${CMAKE_MATCH_1}")
          if(ABI_BYTE_ORDER STREQUAL "NOTFOUND")
            # Tentatively use the value because this is the first occurrence.
            set(ABI_BYTE_ORDER "${byte_order}")
          elseif(NOT ABI_BYTE_ORDER STREQUAL "${byte_order}")
            # Drop value because multiple occurrences do not match.
            set(ABI_BYTE_ORDER "")
          endif()
        endif()
        if("${info}" MATCHES "INFO:abi\\[([^]]*)\\]" AND NOT ABI_NAME)
          set(ABI_NAME "${CMAKE_MATCH_1}")
        endif()
        if("${info}" MATCHES "INFO:arch\\[([^]\"]*)\\]")
          list(APPEND ARCHITECTURE_ID "${CMAKE_MATCH_1}")
        endif()
      endforeach()

      if(ABI_SIZEOF_DPTR)
        set(CMAKE_${lang}_SIZEOF_DATA_PTR "${ABI_SIZEOF_DPTR}" PARENT_SCOPE)
      elseif(CMAKE_${lang}_SIZEOF_DATA_PTR_DEFAULT)
        set(CMAKE_${lang}_SIZEOF_DATA_PTR "${CMAKE_${lang}_SIZEOF_DATA_PTR_DEFAULT}" PARENT_SCOPE)
      endif()

      if(ABI_BYTE_ORDER)
        set(CMAKE_${lang}_BYTE_ORDER "${ABI_BYTE_ORDER}" PARENT_SCOPE)
      endif()

      if(ABI_NAME)
        set(CMAKE_${lang}_COMPILER_ABI "${ABI_NAME}" PARENT_SCOPE)
      endif()

      # The GNU Fortran compiler does not predefine architecture macros.
      if(NOT CMAKE_${lang}_COMPILER_ARCHITECTURE_ID AND NOT ARCHITECTURE_ID
         AND lang STREQUAL "Fortran" AND CMAKE_${lang}_COMPILER_ID STREQUAL "GNU")
        execute_process(COMMAND "${CMAKE_${lang}_COMPILER}" -dumpmachine
          OUTPUT_VARIABLE _dumpmachine_triple OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_VARIABLE  _dumpmachine_stderr
          RESULT_VARIABLE _dumpmachine_result
          )
        if(_dumpmachine_result EQUAL 0)
          include(Internal/CMakeParseCompilerArchitectureId)
          cmake_parse_compiler_architecture_id("${_dumpmachine_triple}" ARCHITECTURE_ID)
        endif()
      endif()

      # For some compilers we detect the architecture id during compiler identification.
      # If this was not one of those, use what was detected during compiler ABI detection,
      # which might be a list, e.g., when CMAKE_OSX_ARCHITECTURES has multiple values.
      if(NOT CMAKE_${lang}_COMPILER_ARCHITECTURE_ID AND ARCHITECTURE_ID)
        list(SORT ARCHITECTURE_ID)
        set(CMAKE_${lang}_COMPILER_ARCHITECTURE_ID "${ARCHITECTURE_ID}" PARENT_SCOPE)
      endif()

      # Parse implicit include directory for this language, if available.
      if(CMAKE_${lang}_VERBOSE_FLAG)
        set (implicit_incdirs "")
        cmake_parse_implicit_include_info("${OUTPUT}" "${lang}"
          implicit_incdirs log rv)
        message(CONFIGURE_LOG
          "Parsed ${lang} implicit include dir info: rv=${rv}\n${log}\n\n")
        if("${rv}" STREQUAL "done")
          # Entries that we have been told to explicitly pass as standard include
          # directories will not be implicitly added by the compiler.
          if(CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES)
            list(REMOVE_ITEM implicit_incdirs ${CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES})
          endif()

          # We parsed implicit include directories, so override the default initializer.
          set(_CMAKE_${lang}_IMPLICIT_INCLUDE_DIRECTORIES_INIT "${implicit_incdirs}")
        endif()
      endif()
      set(CMAKE_${lang}_IMPLICIT_INCLUDE_DIRECTORIES "${_CMAKE_${lang}_IMPLICIT_INCLUDE_DIRECTORIES_INIT}" PARENT_SCOPE)

      if(_CMAKE_${lang}_IMPLICIT_LINK_INFORMATION_DETERMINED_EARLY)
        # Use implicit linker information detected during compiler id step.
        set(implicit_dirs "${CMAKE_${lang}_IMPLICIT_LINK_DIRECTORIES}")
        set(implicit_objs "")
        set(implicit_libs "${CMAKE_${lang}_IMPLICIT_LINK_LIBRARIES}")
        set(implicit_fwks "${CMAKE_${lang}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES}")
      else()
      # Parse implicit linker information for this language, if available.
      set(implicit_dirs "")
      set(implicit_objs "")
      set(implicit_libs "")
      set(implicit_fwks "")
      set(compute_artifacts COMPUTE_LINKER linker_tool)
      if(CMAKE_${lang}_VERBOSE_FLAG)
        list(APPEND compute_artifacts COMPUTE_IMPLICIT_LIBS implicit_libs
                                      COMPUTE_IMPLICIT_DIRS implicit_dirs
                                      COMPUTE_IMPLICIT_FWKS implicit_fwks
                                      COMPUTE_IMPLICIT_OBJECTS implicit_objs)
      endif()
      cmake_parse_implicit_link_info2("${OUTPUT}" log "${CMAKE_${lang}_IMPLICIT_OBJECT_REGEX}"
        ${compute_artifacts} LANGUAGE ${lang})
      message(CONFIGURE_LOG
          "Parsed ${lang} implicit link information:\n${log}\n\n")
      # for VS IDE Intel Fortran we have to figure out the
      # implicit link path for the fortran run time using
      # a try-compile
      if("${lang}" MATCHES "Fortran"
          AND "${CMAKE_GENERATOR}" MATCHES "Visual Studio")
        message(CHECK_START "Determine Intel Fortran Compiler Implicit Link Path")
        # Build a sample project which reports symbols.
        try_compile(IFORT_LIB_PATH_COMPILED
          PROJECT IntelFortranImplicit
          SOURCE_DIR ${CMAKE_ROOT}/Modules/IntelVSImplicitPath
          BINARY_DIR ${CMAKE_BINARY_DIR}/CMakeFiles/IntelVSImplicitPath
          CMAKE_FLAGS
          "-DCMAKE_Fortran_FLAGS:STRING=${CMAKE_Fortran_FLAGS}"
          OUTPUT_VARIABLE _output)
        file(WRITE
          "${CMAKE_BINARY_DIR}/CMakeFiles/IntelVSImplicitPath/output.txt"
          "${_output}")
        include(${CMAKE_BINARY_DIR}/CMakeFiles/IntelVSImplicitPath/output.cmake OPTIONAL)
        message(CHECK_PASS "done")
      endif()
      endif()

      # Implicit link libraries cannot be used explicitly for multiple
      # OS X architectures, so we skip it.
      if(DEFINED CMAKE_OSX_ARCHITECTURES)
        if("${CMAKE_OSX_ARCHITECTURES}" MATCHES ";")
          set(implicit_libs "")
        endif()
      endif()

      # Filter out implicit link directories excluded by our Platform/<os>* modules.
      if(DEFINED CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES_EXCLUDE)
        list(REMOVE_ITEM implicit_dirs ${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES_EXCLUDE})
      endif()

      # Filter out implicit link information excluded by the environment.
      if(DEFINED ENV{CMAKE_${lang}_IMPLICIT_LINK_LIBRARIES_EXCLUDE})
        list(REMOVE_ITEM implicit_libs $ENV{CMAKE_${lang}_IMPLICIT_LINK_LIBRARIES_EXCLUDE})
      endif()
      if(DEFINED ENV{CMAKE_${lang}_IMPLICIT_LINK_DIRECTORIES_EXCLUDE})
        list(REMOVE_ITEM implicit_dirs $ENV{CMAKE_${lang}_IMPLICIT_LINK_DIRECTORIES_EXCLUDE})
      endif()

      set(CMAKE_${lang}_COMPILER_LINKER "${linker_tool}" PARENT_SCOPE)
      cmake_determine_linker_id(${lang} "${linker_tool}")
      set(CMAKE_${lang}_COMPILER_LINKER_ID "${CMAKE_${lang}_COMPILER_LINKER_ID}" PARENT_SCOPE)
      set(CMAKE_${lang}_COMPILER_LINKER_VERSION ${CMAKE_${lang}_COMPILER_LINKER_VERSION} PARENT_SCOPE)
      set(CMAKE_${lang}_COMPILER_LINKER_FRONTEND_VARIANT ${CMAKE_${lang}_COMPILER_LINKER_FRONTEND_VARIANT} PARENT_SCOPE)

      set(CMAKE_${lang}_IMPLICIT_LINK_LIBRARIES "${implicit_libs}" PARENT_SCOPE)
      set(CMAKE_${lang}_IMPLICIT_LINK_DIRECTORIES "${implicit_dirs}" PARENT_SCOPE)
      set(CMAKE_${lang}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "${implicit_fwks}" PARENT_SCOPE)

      cmake_parse_library_architecture(${lang} "${implicit_dirs}" "${implicit_objs}" architecture_flag)
      if(architecture_flag)
        set(CMAKE_${lang}_LIBRARY_ARCHITECTURE "${architecture_flag}" PARENT_SCOPE)
      endif()

    else()
      message(CHECK_FAIL "failed")
    endif()
  endif()
endfunction()
