# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# CPack script for creating Debian package
# Author: Mathieu Malaterre
#
# http://wiki.debian.org/HowToPackageForDebian

if(CMAKE_BINARY_DIR)
  message(FATAL_ERROR "CPackDeb.cmake may only be used by CPack internally.")
endif()

function(cpack_deb_variable_fallback OUTPUT_VAR_NAME)
  set(FALLBACK_VAR_NAMES ${ARGN})

  foreach(variable_name IN LISTS FALLBACK_VAR_NAMES)
    if(${variable_name})
      set(${OUTPUT_VAR_NAME} "${${variable_name}}" PARENT_SCOPE)
      break()
    endif()
  endforeach()
endfunction()

function(get_component_package_name var component)
  string(TOUPPER "${component}" component_upcase)
  if(CPACK_DEBIAN_${component_upcase}_PACKAGE_NAME)
    string(TOLOWER "${CPACK_DEBIAN_${component_upcase}_PACKAGE_NAME}" package_name)
  else()
    string(TOLOWER "${CPACK_DEBIAN_PACKAGE_NAME}-${component}" package_name)
  endif()

  set("${var}" "${package_name}" PARENT_SCOPE)
endfunction()

#extract library name and version for given shared object
function(extract_so_info shared_object libname version)
  if(CPACK_READELF_EXECUTABLE)
    execute_process(COMMAND "${CPACK_READELF_EXECUTABLE}" -d "${shared_object}"
      WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(result EQUAL 0)
      string(REGEX MATCH "\\(?SONAME\\)?[^\n]*\\[([^\n]+)\\.so\\.([^\n]*)\\]" soname "${output}")
      set(${libname} "${CMAKE_MATCH_1}" PARENT_SCOPE)
      set(${version} "${CMAKE_MATCH_2}" PARENT_SCOPE)
    else()
      message(WARNING "Error running readelf for \"${shared_object}\"")
    endif()
  else()
    message(FATAL_ERROR "Readelf utility is not available.")
  endif()
endfunction()

#extract RUNPATH and RPATH for given shared object or executable
function(extract_runpath_and_rpath shared_object_or_executable runpath rpath)
  if(CPACK_READELF_EXECUTABLE)
    execute_process(COMMAND "${CPACK_READELF_EXECUTABLE}" -d "${shared_object_or_executable}"
      WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(result EQUAL 0)
      string(REGEX MATCH "\\(?RUNPATH\\)?[^\n]*\\[([^\n]+)\\]" found_runpath "${output}")
      string(REPLACE ":" ";" found_runpath "${CMAKE_MATCH_1}")
      list(REMOVE_DUPLICATES found_runpath)
      string(REGEX MATCH "\\(?RPATH\\)?[^\n]*\\[([^\n]+)\\]"   found_rpath   "${output}")
      string(REPLACE ":" ";" found_rpath   "${CMAKE_MATCH_1}")
      list(REMOVE_DUPLICATES found_rpath)
      set(${runpath} "${found_runpath}" PARENT_SCOPE)
      set(${rpath}   "${found_rpath}"   PARENT_SCOPE)
    else()
      message(WARNING "Error running readelf for \"${shared_object_or_executable}\"")
    endif()
  else()
    message(FATAL_ERROR "Readelf utility is not available.")
  endif()
endfunction()

#sanitizes the given directory name if required
function(get_sanitized_dirname dirname outvar)
  # NOTE: This pattern has to stay in sync with the 'prohibited_chars' variable
  #       defined in the C++ function `CPackGenerator::GetSanitizedDirOrFileName`!
  set(prohibited_chars_pattern "[<]|[>]|[\"]|[/]|[\\]|[|]|[?]|[*]|[`]")
  if("${dirname}" MATCHES "${prohibited_chars_pattern}")
    string(MD5 sanitized_dirname "${dirname}")
    set(${outvar} "${sanitized_dirname}" PARENT_SCOPE)
  else()
    set(${outvar} "${dirname}" PARENT_SCOPE)
  endif()
endfunction()

#retrieve packaging directories of components the current component depends on
# Note: May only be called from within 'cpack_deb_prepare_package_var'!
function(get_packaging_dirs_of_dependencies outvar)
  if(CPACK_DEB_PACKAGE_COMPONENT)
    if(NOT DEFINED WDIR OR NOT DEFINED _local_component_name)
      message(FATAL_ERROR "CPackDeb: Function '${CMAKE_CURRENT_FUNCTION}' not called from correct function scope!")
    endif()
    set(result_list)
    foreach(dependency_name IN LISTS CPACK_COMPONENT_${_local_component_name}_DEPENDS)
      get_sanitized_dirname("${dependency_name}" dependency_name)
      cmake_path(APPEND_STRING WDIR "/../${dependency_name}" OUTPUT_VARIABLE dependency_packaging_dir)
      cmake_path(NORMAL_PATH dependency_packaging_dir)
      list(APPEND result_list "${dependency_packaging_dir}")
    endforeach()
    set(${outvar} "${result_list}" PARENT_SCOPE)  # Set return variable.
  else()
    set(${outvar} "" PARENT_SCOPE)  # Clear return variable.
  endif()
endfunction()

function(cpack_deb_check_description SUMMARY LINES RESULT_VARIABLE)
  set(_result TRUE)

  # Get the summary line
  if(NOT SUMMARY MATCHES "^[^\\s].*$")
    set(_result FALSE)
    set(${RESULT_VARIABLE} ${_result} PARENT_SCOPE)
    return()
  endif()

  foreach(_line IN LISTS LINES)
    if(NOT _line MATCHES "^ +[^ ]+.*$")
      set(_result FALSE)
      break()
    endif()
  endforeach()

  set(${RESULT_VARIABLE} ${_result} PARENT_SCOPE)
endfunction()

function(cpack_deb_format_package_description TEXT OUTPUT_VAR)
  # Turn the possible multi-line string into a list
  string(UUID uuid NAMESPACE 00000000-0000-0000-0000-000000000000 TYPE SHA1)
  string(REPLACE ";" "${uuid}" _text "${TEXT}")
  string(REPLACE "\n" ";" _lines "${_text}")
  list(POP_FRONT _lines _summary)

  # If the description ends with a newline (e.g. typically if it was read
  # from a file) the last line will be empty. We drop it here, otherwise
  # it would be replaced by a `.` which would lead to the package violating
  # the extended-description-contains-empty-paragraph debian policy
  list(POP_BACK _lines _last_line)
  string(STRIP "${_last_line}" _last_line_strip)
  if(_last_line_strip)
    list(APPEND _lines "${_last_line_strip}")
  endif()

  # Check if reformatting required
  cpack_deb_check_description("${_summary}" "${_lines}" _result)
  if(_result)
    # Ok, no formatting required
    set(${OUTPUT_VAR} "${TEXT}" PARENT_SCOPE)
    return()
  endif()

  # Format the summary line
  string(STRIP "${_summary}" _summary)

  # Make sure the rest formatted properly
  set(_result)
  foreach(_line IN LISTS _lines)
    string(STRIP "${_line}" _line_strip)
    if(NOT _line_strip)
      # Replace empty lines w/ a _single full stop character_
      set(_line " .")
    else()
      # Prepend the normal lines w/ a single space.
      # If the line already starts w/ at least one space,
      # it'll become _verbatim_ (assuming it supposed to be
      # verbatim in the original text).
      string(PREPEND _line " ")
    endif()
    list(APPEND _result "${_line}")
  endforeach()

  list(PREPEND _result "${_summary}")
  list(JOIN _result "\n" _result)
  string(REPLACE "${uuid}"  ";" _result "${_result}")
  set(${OUTPUT_VAR} "${_result}" PARENT_SCOPE)
endfunction()

function(cpack_deb_prepare_package_vars)
  # CPACK_DEBIAN_PACKAGE_SHLIBDEPS
  # If specify OFF, only user depends are used
  if(NOT DEFINED CPACK_DEBIAN_PACKAGE_SHLIBDEPS)
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS OFF)
  endif()

  set(WDIR "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}${CPACK_DEB_PACKAGE_COMPONENT_PART_PATH}")
  set(DBGSYMDIR "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}${CPACK_DEB_PACKAGE_COMPONENT_PART_PATH}-dbgsym")
  file(REMOVE_RECURSE "${DBGSYMDIR}")

  # per component automatic discover: some of the component might not have
  # binaries.
  if(CPACK_DEB_PACKAGE_COMPONENT)
    string(TOUPPER "${CPACK_DEB_PACKAGE_COMPONENT}" _local_component_name)
    set(_component_shlibdeps_var "CPACK_DEBIAN_${_local_component_name}_PACKAGE_SHLIBDEPS")

    # if set, overrides the global configuration
    if(DEFINED ${_component_shlibdeps_var})
      set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS "${${_component_shlibdeps_var}}")
      if(CPACK_DEBIAN_PACKAGE_DEBUG)
        message("CPackDeb Debug: component '${CPACK_DEB_PACKAGE_COMPONENT}' dpkg-shlibdeps set to ${CPACK_DEBIAN_PACKAGE_SHLIBDEPS}")
      endif()
    endif()
  endif()

  cpack_deb_variable_fallback("CPACK_DEBIAN_DEBUGINFO_PACKAGE"
    "CPACK_DEBIAN_${_local_component_name}_DEBUGINFO_PACKAGE"
    "CPACK_DEBIAN_DEBUGINFO_PACKAGE")
  if(CPACK_DEBIAN_PACKAGE_SHLIBDEPS OR CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS OR CPACK_DEBIAN_DEBUGINFO_PACKAGE)
    # Generating binary list - Get type of all install files
    file(GLOB_RECURSE FILE_PATHS_ LIST_DIRECTORIES false RELATIVE "${WDIR}" "${WDIR}/*")

    find_program(FILE_EXECUTABLE file)
    if(NOT FILE_EXECUTABLE)
      message(FATAL_ERROR "CPackDeb: file utility is not available. CPACK_DEBIAN_PACKAGE_SHLIBDEPS and CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS options are not available.")
    endif()

    # get file info so that we can determine if file is executable or not
    unset(CPACK_DEB_INSTALL_FILES)
    foreach(FILE_ IN LISTS FILE_PATHS_)
      execute_process(COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${FILE_EXECUTABLE} "./${FILE_}"
        WORKING_DIRECTORY "${WDIR}"
        RESULT_VARIABLE FILE_RESULT_
        OUTPUT_VARIABLE INSTALL_FILE_)
      if(NOT FILE_RESULT_ EQUAL 0)
        message(FATAL_ERROR "CPackDeb: execution of command: '${FILE_EXECUTABLE} ./${FILE_}' failed with exit code: ${FILE_RESULT_}")
      endif()
      list(APPEND CPACK_DEB_INSTALL_FILES "${INSTALL_FILE_}")
    endforeach()

    # Only dynamically linked ELF files are included
    # Extract only file name in front of ":"
    foreach(_FILE IN LISTS CPACK_DEB_INSTALL_FILES)
      if(_FILE MATCHES "ELF.*dynamically linked")
        string(REGEX MATCH "(^.*):" _FILE_NAME "${_FILE}")
        list(APPEND CPACK_DEB_BINARY_FILES "${CMAKE_MATCH_1}")
        set(CONTAINS_EXECUTABLE_FILES_ TRUE)
      endif()
      if(_FILE MATCHES "ELF.*shared object")
        string(REGEX MATCH "(^.*):" _FILE_NAME "${_FILE}")
        list(APPEND CPACK_DEB_SHARED_OBJECT_FILES "${CMAKE_MATCH_1}")
      endif()
      if(_FILE MATCHES "ELF.*not stripped")
        string(REGEX MATCH "(^.*):" _FILE_NAME "${_FILE}")
        list(APPEND CPACK_DEB_UNSTRIPPED_FILES "${CMAKE_MATCH_1}")
      endif()
    endforeach()
  endif()

  find_program(CPACK_READELF_EXECUTABLE NAMES readelf)

  if(CPACK_DEBIAN_DEBUGINFO_PACKAGE AND CPACK_DEB_UNSTRIPPED_FILES)
    find_program(CPACK_OBJCOPY_EXECUTABLE NAMES objcopy)

    if(NOT CPACK_OBJCOPY_EXECUTABLE)
      message(FATAL_ERROR "debuginfo packages require the objcopy tool")
    endif()
    if(NOT CPACK_READELF_EXECUTABLE)
      message(FATAL_ERROR "debuginfo packages require the readelf tool")
    endif()

    file(RELATIVE_PATH _DBGSYM_ROOT "${CPACK_TEMPORARY_DIRECTORY}" "${DBGSYMDIR}")
    foreach(_FILE IN LISTS CPACK_DEB_UNSTRIPPED_FILES)

      # Get the file's Build ID
      execute_process(COMMAND env LC_ALL=C ${CPACK_READELF_EXECUTABLE} -n "${_FILE}"
        WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
        OUTPUT_VARIABLE READELF_OUTPUT
        RESULT_VARIABLE READELF_RESULT
        ERROR_VARIABLE READELF_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE )
      if(NOT READELF_RESULT EQUAL 0)
        message(FATAL_ERROR "CPackDeb: readelf: '${READELF_ERROR}';\n"
            "executed command: '${CPACK_READELF_EXECUTABLE} -n ${_FILE}'")
      endif()
      if(READELF_OUTPUT MATCHES "Build ID: ([0-9a-zA-Z][0-9a-zA-Z])([0-9a-zA-Z]*)")
        set(_BUILD_ID_START ${CMAKE_MATCH_1})
        set(_BUILD_ID_REMAINING ${CMAKE_MATCH_2})
        list(APPEND BUILD_IDS ${_BUILD_ID_START}${_BUILD_ID_REMAINING})
      else()
        message(FATAL_ERROR "Unable to determine Build ID for ${_FILE}")
      endif()

      # Split out the debug symbols from the binaries
      set(_FILE_DBGSYM ${_DBGSYM_ROOT}/usr/lib/debug/.build-id/${_BUILD_ID_START}/${_BUILD_ID_REMAINING}.debug)
      get_filename_component(_OUT_DIR "${_FILE_DBGSYM}" DIRECTORY)
      file(MAKE_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}/${_OUT_DIR}")
      execute_process(COMMAND ${CPACK_OBJCOPY_EXECUTABLE} --only-keep-debug "${_FILE}" "${_FILE_DBGSYM}"
        WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
        OUTPUT_VARIABLE OBJCOPY_OUTPUT
        RESULT_VARIABLE OBJCOPY_RESULT
        ERROR_VARIABLE OBJCOPY_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE )
      if(NOT OBJCOPY_RESULT EQUAL 0)
        message(FATAL_ERROR "CPackDeb: objcopy: '${OBJCOPY_ERROR}';\n"
            "executed command: '${CPACK_OBJCOPY_EXECUTABLE} --only-keep-debug ${_FILE} ${_FILE_DBGSYM}'")
      endif()
      execute_process(COMMAND ${CPACK_OBJCOPY_EXECUTABLE} --strip-unneeded ${_FILE}
        WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
        OUTPUT_VARIABLE OBJCOPY_OUTPUT
        RESULT_VARIABLE OBJCOPY_RESULT
        ERROR_VARIABLE OBJCOPY_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE )
      if(NOT OBJCOPY_RESULT EQUAL 0)
        message(FATAL_ERROR "CPackDeb: objcopy: '${OBJCOPY_ERROR}';\n"
            "executed command: '${CPACK_OBJCOPY_EXECUTABLE} --strip-debug ${_FILE}'")
      endif()
      execute_process(COMMAND ${CPACK_OBJCOPY_EXECUTABLE} --add-gnu-debuglink=${_FILE_DBGSYM} ${_FILE}
        WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
        OUTPUT_VARIABLE OBJCOPY_OUTPUT
        RESULT_VARIABLE OBJCOPY_RESULT
        ERROR_VARIABLE OBJCOPY_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE )
      if(NOT OBJCOPY_RESULT EQUAL 0)
        message(FATAL_ERROR "CPackDeb: objcopy: '${OBJCOPY_ERROR}';\n"
            "executed command: '${CPACK_OBJCOPY_EXECUTABLE} --add-gnu-debuglink=${_FILE_DBGSYM} ${_FILE}'")
      endif()
    endforeach()
  endif()

  if(CPACK_DEBIAN_PACKAGE_SHLIBDEPS)
    # dpkg-shlibdeps is a Debian utility for generating dependency list
    find_program(SHLIBDEPS_EXECUTABLE dpkg-shlibdeps)

    if(SHLIBDEPS_EXECUTABLE)
      # Check version of the dpkg-shlibdeps tool using CPackDEB method
      execute_process(COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${SHLIBDEPS_EXECUTABLE} --version
        OUTPUT_VARIABLE _TMP_VERSION
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(_TMP_VERSION MATCHES "dpkg-shlibdeps version ([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(SHLIBDEPS_EXECUTABLE_VERSION "${CMAKE_MATCH_1}")
      else()
        unset(SHLIBDEPS_EXECUTABLE_VERSION)
      endif()

      if(CPACK_DEBIAN_PACKAGE_DEBUG)
        message("CPackDeb Debug: dpkg-shlibdeps --version output is '${_TMP_VERSION}'")
        message("CPackDeb Debug: dpkg-shlibdeps version is <${SHLIBDEPS_EXECUTABLE_VERSION}>")
      endif()

      if(CONTAINS_EXECUTABLE_FILES_)
        message("CPackDeb: - Generating dependency list")

        # Create blank control file for running dpkg-shlibdeps
        # There might be some other way to invoke dpkg-shlibdeps without creating this file
        # but standard debian package should not have anything that can collide with this file or directory
        file(MAKE_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}/debian)
        file(WRITE ${CPACK_TEMPORARY_DIRECTORY}/debian/control "")

        # Create a DEBIAN directory so that dpkg-shlibdeps can find the package dir when resolving $ORIGIN.
        file(MAKE_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}/DEBIAN")

        # Add --ignore-missing-info if the tool supports it
        execute_process(COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${SHLIBDEPS_EXECUTABLE} --help
          OUTPUT_VARIABLE _TMP_HELP
          ERROR_QUIET
          OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_TMP_HELP MATCHES "--ignore-missing-info")
          set(IGNORE_MISSING_INFO_FLAG "--ignore-missing-info")
        endif()

        # Add -l option if the tool supports it?
        if(DEFINED SHLIBDEPS_EXECUTABLE_VERSION AND SHLIBDEPS_EXECUTABLE_VERSION VERSION_GREATER_EQUAL 1.17.0)
          unset(PRIVATE_SEARCH_DIR_OPTIONS)

          # Use directories provided via CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS
          if(NOT "${CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS}" STREQUAL "")
            foreach(path IN LISTS CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS)
              cmake_path(NORMAL_PATH path)  # Required for dpkg-shlibdeps!
              list(APPEND PRIVATE_SEARCH_DIR_OPTIONS "-l${path}")
            endforeach()
          endif()

          # Use directories extracted from RUNPATH/RPATH
          get_packaging_dirs_of_dependencies(deps_packaging_dirs)
          foreach(exe IN LISTS CPACK_DEB_BINARY_FILES)
            cmake_path(GET exe PARENT_PATH exe_dir)
            extract_runpath_and_rpath(${exe} runpath rpath)
            # If RUNPATH is available, RPATH will be ignored. Therefore we have to do the same here!
            if (NOT "${runpath}" STREQUAL "")
              set(selected_rpath "${runpath}")
            else()
              set(selected_rpath "${rpath}")
            endif()
            foreach(search_path IN LISTS selected_rpath)
              if ("${search_path}" MATCHES "^[$]ORIGIN" OR "${search_path}" MATCHES "^[$][{]ORIGIN[}]")
                foreach(deps_pkgdir IN LISTS deps_packaging_dirs)
                  string(REPLACE "\$ORIGIN" "${deps_pkgdir}/${exe_dir}" path "${search_path}")
                  string(REPLACE "\${ORIGIN}" "${deps_pkgdir}/${exe_dir}" path "${path}")
                  cmake_path(NORMAL_PATH path)  # Required for dpkg-shlibdeps!
                  list(APPEND PRIVATE_SEARCH_DIR_OPTIONS "-l${path}")
                endforeach()
              endif()
            endforeach()
          endforeach()

          list(REMOVE_DUPLICATES PRIVATE_SEARCH_DIR_OPTIONS)
        elseif(NOT "${CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS}" STREQUAL "")
          message(WARNING "CPackDeb: dkpg-shlibdeps is too old. \"CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS\" is therefore ignored.")
        endif()

        # Execute dpkg-shlibdeps
        # --ignore-missing-info : allow dpkg-shlibdeps to run even if some libs do not belong to a package
        # -l<dir>: make dpkg-shlibdeps also search in this directory for (private) shared library dependencies
        # -O : print to STDOUT
        execute_process(COMMAND ${SHLIBDEPS_EXECUTABLE} ${PRIVATE_SEARCH_DIR_OPTIONS} ${IGNORE_MISSING_INFO_FLAG} -O ${CPACK_DEB_BINARY_FILES}
          WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
          OUTPUT_VARIABLE SHLIBDEPS_OUTPUT
          RESULT_VARIABLE SHLIBDEPS_RESULT
          ERROR_VARIABLE SHLIBDEPS_ERROR
          OUTPUT_STRIP_TRAILING_WHITESPACE )

        # E2K OSL 6.0.1 and prior has broken dpkg-shlibdeps. CPack will deal with that (mocking SHLIBDEPS_OUTPUT), but inform user of this.
        if("${SHLIBDEPS_ERROR}" MATCHES "unknown gcc system type e2k.*, falling back to default")
          message(WARNING "CPackDeb: broken dpkg-shlibdeps on E2K detected, will fall back to minimal dependencies.\n"
                  "You should expect that dependencies list in the package will be incomplete.")
          set(SHLIBDEPS_OUTPUT "shlibs:Depends=libc6, lcc-libs")
        endif()

        if(CPACK_DEBIAN_PACKAGE_DEBUG)
          # dpkg-shlibdeps will throw some warnings if some input files are not binary
          message( "CPackDeb Debug: dpkg-shlibdeps warnings \n${SHLIBDEPS_ERROR}")
        endif()
        if(NOT SHLIBDEPS_RESULT EQUAL 0)
          message(FATAL_ERROR "CPackDeb: dpkg-shlibdeps: '${SHLIBDEPS_ERROR}';\n"
              "executed command: '${SHLIBDEPS_EXECUTABLE} ${PRIVATE_SEARCH_DIR_OPTIONS} ${IGNORE_MISSING_INFO_FLAG} -O ${CPACK_DEB_BINARY_FILES}';\n"
              "found files: '${INSTALL_FILE_}';\n"
              "files info: '${CPACK_DEB_INSTALL_FILES}';\n"
              "binary files: '${CPACK_DEB_BINARY_FILES}'")
        endif()

        #Get rid of prefix generated by dpkg-shlibdeps
        string(REGEX REPLACE "^.*Depends=" "" CPACK_DEBIAN_PACKAGE_AUTO_DEPENDS "${SHLIBDEPS_OUTPUT}")

        if(CPACK_DEBIAN_PACKAGE_DEBUG)
          message("CPackDeb Debug: Found dependency: ${CPACK_DEBIAN_PACKAGE_AUTO_DEPENDS} from output ${SHLIBDEPS_OUTPUT}")
        endif()

        # Remove blank control file
        # Might not be safe if package actual contain file or directory named debian
        file(REMOVE_RECURSE "${CPACK_TEMPORARY_DIRECTORY}/debian")

        # remove temporary directory that was created only for dpkg-shlibdeps execution
        file(REMOVE_RECURSE "${CPACK_TEMPORARY_DIRECTORY}/DEBIAN")
      else()
        if(CPACK_DEBIAN_PACKAGE_DEBUG)
          message(AUTHOR_WARNING "CPackDeb Debug: Using only user-provided depends because package does not contain executable files that link to shared libraries.")
        endif()
      endif()
    else()
      message("CPackDeb: Using only user-provided dependencies because dpkg-shlibdeps is not found.")
    endif()

  else()
    if(CPACK_DEBIAN_PACKAGE_DEBUG)
      message("CPackDeb Debug: Using only user-provided dependencies")
    endif()
  endif()

  # Let's define the control file found in debian package:

  # Binary package:
  # http://www.debian.org/doc/debian-policy/ch-controlfields.html#s-binarycontrolfiles

  # DEBIAN/control
  # debian policy enforce lower case for package name
  # Package: (mandatory)
  if(NOT CPACK_DEBIAN_PACKAGE_NAME)
    string(TOLOWER "${CPACK_PACKAGE_NAME}" CPACK_DEBIAN_PACKAGE_NAME)
  endif()

  # Version: (mandatory)
  if(NOT CPACK_DEBIAN_PACKAGE_VERSION)
    if(NOT CPACK_PACKAGE_VERSION)
      message(FATAL_ERROR "CPackDeb: Debian package requires a package version")
    endif()
    set(CPACK_DEBIAN_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
  endif()

  if(DEFINED CPACK_DEBIAN_PACKAGE_RELEASE OR DEFINED CPACK_DEBIAN_PACKAGE_EPOCH)
    # only test the version format if CPACK_DEBIAN_PACKAGE_RELEASE or
    # CPACK_DEBIAN_PACKAGE_EPOCH is set
    if(NOT CPACK_DEBIAN_PACKAGE_VERSION MATCHES "^[0-9][A-Za-z0-9.+~-]*$")
      message(FATAL_ERROR
        "CPackDeb: Debian package version must confirm to \"^[0-9][A-Za-z0-9.+~-]*$\" regex!")
    endif()
  else()
    # before CMake 3.10 version format was not tested so only warn to preserve
    # backward compatibility
    if(NOT CPACK_DEBIAN_PACKAGE_VERSION MATCHES "^([0-9]+:)?[0-9][A-Za-z0-9.+~-]*$")
      message(AUTHOR_WARNING
        "CPackDeb: Debian package versioning ([<epoch>:]<version>[-<release>])"
        " should confirm to \"^([0-9]+:)?[0-9][A-Za-z0-9.+~-]*$\" regex in"
        " order to satisfy Debian packaging rules.")
    endif()
  endif()

  if(CPACK_DEBIAN_PACKAGE_RELEASE)
    if(NOT CPACK_DEBIAN_PACKAGE_RELEASE MATCHES "^[A-Za-z0-9.+~]+$")
      message(FATAL_ERROR
        "CPackDeb: Debian package release must confirm to \"^[A-Za-z0-9.+~]+$\" regex!")
    endif()
    string(APPEND CPACK_DEBIAN_PACKAGE_VERSION
      "-${CPACK_DEBIAN_PACKAGE_RELEASE}")
  elseif(DEFINED CPACK_DEBIAN_PACKAGE_EPOCH)
    # only test the version format if CPACK_DEBIAN_PACKAGE_RELEASE or
    # CPACK_DEBIAN_PACKAGE_EPOCH is set - versions CPack/Deb generator before
    # CMake 3.10 did not check for version format so we have to preserve
    # backward compatibility
    if(CPACK_DEBIAN_PACKAGE_VERSION MATCHES ".*-.*")
      message(FATAL_ERROR
        "CPackDeb: Debian package version must not contain hyphens when CPACK_DEBIAN_PACKAGE_RELEASE is not provided!")
    endif()
  endif()

  if(CPACK_DEBIAN_PACKAGE_EPOCH)
    if(NOT CPACK_DEBIAN_PACKAGE_EPOCH MATCHES "^[0-9]+$")
      message(FATAL_ERROR
        "CPackDeb: Debian package epoch must confirm to \"^[0-9]+$\" regex!")
    endif()
    set(CPACK_DEBIAN_PACKAGE_VERSION
      "${CPACK_DEBIAN_PACKAGE_EPOCH}:${CPACK_DEBIAN_PACKAGE_VERSION}")
  endif()

  # Architecture: (mandatory)
  if(CPACK_DEB_PACKAGE_COMPONENT AND CPACK_DEBIAN_${_local_component_name}_PACKAGE_ARCHITECTURE)
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "${CPACK_DEBIAN_${_local_component_name}_PACKAGE_ARCHITECTURE}")
  elseif(NOT CPACK_DEBIAN_PACKAGE_ARCHITECTURE)
    # There is no such thing as i686 architecture on debian, you should use i386 instead
    # $ dpkg --print-architecture
    find_program(DPKG_CMD dpkg)
    if(NOT DPKG_CMD)
      message(STATUS "CPackDeb: Can not find dpkg in your path, default to i386.")
      set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE i386)
    endif()
    execute_process(COMMAND "${DPKG_CMD}" --print-architecture
      OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
  endif()

  # Source: (optional)
  # in case several packages are constructed from a unique source
  # (multipackaging), the source may be indicated as well.
  # The source might contain a version if the generated package
  # version is different from the source version
  if(NOT CPACK_DEBIAN_PACKAGE_SOURCE)
    set(CPACK_DEBIAN_PACKAGE_SOURCE "")
  endif()

  # have a look at get_property(result GLOBAL PROPERTY ENABLED_FEATURES),
  # this returns the successful find_package() calls, maybe this can help
  # Depends:
  # You should set: DEBIAN_PACKAGE_DEPENDS
  # TODO: automate 'objdump -p | grep NEEDED'

  # if per-component variable, overrides the global CPACK_DEBIAN_PACKAGE_${variable_type_}
  # automatic dependency discovery will be performed afterwards.
  if(CPACK_DEB_PACKAGE_COMPONENT)
    foreach(value_type_ IN ITEMS DEPENDS RECOMMENDS SUGGESTS PREDEPENDS ENHANCES BREAKS CONFLICTS PROVIDES REPLACES MULTIARCH SOURCE SECTION PRIORITY NAME)
      set(_component_var "CPACK_DEBIAN_${_local_component_name}_PACKAGE_${value_type_}")

      # if set, overrides the global variable
      if(DEFINED ${_component_var})
        set(CPACK_DEBIAN_PACKAGE_${value_type_} "${${_component_var}}")
        if(CPACK_DEBIAN_PACKAGE_DEBUG)
          message("CPackDeb Debug: component '${CPACK_DEB_PACKAGE_COMPONENT}' ${value_type_} "
            "value set to '${CPACK_DEBIAN_PACKAGE_${value_type_}}'")
        endif()
      endif()
    endforeach()

    if(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS)
      unset(COMPONENT_DEPENDS)
      foreach(_PACK IN LISTS CPACK_COMPONENT_${_local_component_name}_DEPENDS)
        get_component_package_name(_PACK_NAME "${_PACK}")
        list(PREPEND COMPONENT_DEPENDS "${_PACK_NAME} (= ${CPACK_DEBIAN_PACKAGE_VERSION})")
      endforeach()
      list(JOIN COMPONENT_DEPENDS ", " COMPONENT_DEPENDS)
      if(COMPONENT_DEPENDS)
        list(PREPEND CPACK_DEBIAN_PACKAGE_DEPENDS ${COMPONENT_DEPENDS})
      endif()
    endif()
  endif()

  # at this point, the CPACK_DEBIAN_PACKAGE_DEPENDS is properly set
  # to the minimal dependency of the package
  # Append automatically discovered dependencies .
  if(CPACK_DEBIAN_PACKAGE_AUTO_DEPENDS)
    list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_AUTO_DEPENDS})
  endif()

  list(JOIN CPACK_DEBIAN_PACKAGE_DEPENDS ", " CPACK_DEBIAN_PACKAGE_DEPENDS)
  if(NOT CPACK_DEBIAN_PACKAGE_DEPENDS)
    message(STATUS "CPACK_DEBIAN_PACKAGE_DEPENDS not set, the package will have no dependencies.")
  endif()

  # Maintainer: (mandatory)
  if(NOT CPACK_DEBIAN_PACKAGE_MAINTAINER)
    if(NOT CPACK_PACKAGE_CONTACT)
      message(FATAL_ERROR "CPackDeb: Debian package requires a maintainer for a package, set CPACK_PACKAGE_CONTACT or CPACK_DEBIAN_PACKAGE_MAINTAINER")
    endif()
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
  endif()

  # Description: (mandatory)
  # Try package description first
  if(CPACK_USED_DEFAULT_PACKAGE_DESCRIPTION_FILE)
    set(_desc_fallback)
  else()
    set(_desc_fallback "CPACK_PACKAGE_DESCRIPTION")
  endif()
  if(CPACK_DEB_PACKAGE_COMPONENT)
    cpack_deb_variable_fallback("CPACK_DEBIAN_PACKAGE_DESCRIPTION"
      "CPACK_DEBIAN_${_local_component_name}_DESCRIPTION"
      "CPACK_COMPONENT_${_local_component_name}_DESCRIPTION")
  else()
    cpack_deb_variable_fallback("CPACK_DEBIAN_PACKAGE_DESCRIPTION"
      "CPACK_DEBIAN_PACKAGE_DESCRIPTION"
      ${_desc_fallback})
  endif()

  # Still no description? ... and description file has set ...
  if(NOT CPACK_DEBIAN_PACKAGE_DESCRIPTION
     AND CPACK_PACKAGE_DESCRIPTION_FILE
     AND NOT CPACK_PACKAGE_DESCRIPTION_FILE STREQUAL CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE)
    # Read `CPACK_PACKAGE_DESCRIPTION_FILE` then...
    file(READ ${CPACK_PACKAGE_DESCRIPTION_FILE} CPACK_DEBIAN_PACKAGE_DESCRIPTION)
  endif()

  # Still no description? #2
  if(NOT CPACK_DEBIAN_PACKAGE_DESCRIPTION)
    # Try to get `CPACK_PACKAGE_DESCRIPTION_SUMMARY` as the last hope
    if(CPACK_PACKAGE_DESCRIPTION_SUMMARY)
      set(CPACK_DEBIAN_PACKAGE_DESCRIPTION ${CPACK_PACKAGE_DESCRIPTION_SUMMARY})
    else()
      # Giving up! Report an error...
      set(_description_failure_message
        "CPackDeb: Debian package requires a summary for a package, set CPACK_PACKAGE_DESCRIPTION_SUMMARY or CPACK_DEBIAN_PACKAGE_DESCRIPTION")
      if(CPACK_DEB_PACKAGE_COMPONENT)
        string(APPEND _description_failure_message
          " or CPACK_DEBIAN_${_local_component_name}_DESCRIPTION")
      endif()
      message(FATAL_ERROR "${_description_failure_message}")
    endif()

  # Ok, description has set. According to the `Debian Policy Manual`_ the first
  # line is a package summary.  Try to get it as well...
  # See also: https://www.debian.org/doc/debian-policy/ch-controlfields.html#description
  elseif(CPACK_PACKAGE_DESCRIPTION_SUMMARY AND
         NOT CPACK_PACKAGE_DESCRIPTION_SUMMARY STREQUAL CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY)
    # Merge summary w/ the detailed description
    string(PREPEND CPACK_DEBIAN_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}\n")
  endif()
  # assert(CPACK_DEBIAN_PACKAGE_DESCRIPTION)

  # Make sure description is properly formatted
  cpack_deb_format_package_description(
    "${CPACK_DEBIAN_PACKAGE_DESCRIPTION}"
    CPACK_DEBIAN_PACKAGE_DESCRIPTION
  )

  # Homepage: (optional)
  if(NOT CPACK_DEBIAN_PACKAGE_HOMEPAGE AND CPACK_PACKAGE_HOMEPAGE_URL)
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "${CPACK_PACKAGE_HOMEPAGE_URL}")
  endif()

  # Section: (recommended)
  if(NOT CPACK_DEBIAN_PACKAGE_SECTION)
    set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
  endif()

  # Priority: (recommended)
  if(NOT CPACK_DEBIAN_PACKAGE_PRIORITY)
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
  endif()

  if(CPACK_DEBIAN_ARCHIVE_TYPE)
    if(CPACK_DEBIAN_ARCHIVE_TYPE STREQUAL "paxr")
      message(DEPRECATION "CPACK_DEBIAN_ARCHIVE_TYPE set to old and invalid "
        "type 'paxr', mapping to 'gnutar'")
      set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")
    elseif(NOT CPACK_DEBIAN_ARCHIVE_TYPE STREQUAL "gnutar")
      message(FATAL_ERROR "CPACK_DEBIAN_ARCHIVE_TYPE set to unsupported"
        "type ${CPACK_DEBIAN_ARCHIVE_TYPE}")
    endif()
  else()
    set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")
  endif()

  # Compression: (recommended)
  if(NOT CPACK_DEBIAN_COMPRESSION_TYPE)
    set(CPACK_DEBIAN_COMPRESSION_TYPE "gzip")
  endif()

  # Recommends:
  # You should set: CPACK_DEBIAN_PACKAGE_RECOMMENDS

  # Suggests:
  # You should set: CPACK_DEBIAN_PACKAGE_SUGGESTS

  # CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
  # This variable allow advanced user to add custom script to the control.tar.gz (inside the .deb archive)
  # Typical examples are:
  # - conffiles
  # - postinst
  # - postrm
  # - prerm
  # Usage:
  # set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
  #    "${CMAKE_CURRENT_SOURCE_DIR}/prerm;${CMAKE_CURRENT_SOURCE_DIR}/postrm")

  # Are we packaging components ?
  if(CPACK_DEB_PACKAGE_COMPONENT)
    # override values with per component version if set
    foreach(VAR_NAME_ IN ITEMS PACKAGE_CONTROL_EXTRA PACKAGE_CONTROL_STRICT_PERMISSION)
      if(CPACK_DEBIAN_${_local_component_name}_${VAR_NAME_})
        set(CPACK_DEBIAN_${VAR_NAME_} "${CPACK_DEBIAN_${_local_component_name}_${VAR_NAME_}}")
      endif()
    endforeach()
    get_component_package_name(CPACK_DEBIAN_PACKAGE_NAME ${_local_component_name})
  endif()

  if(NOT CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS_POLICY)
    set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS_POLICY "=")
  endif()

  unset(CPACK_DEBIAN_PACKAGE_SHLIBS_LIST)

  if(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS)
    if(CPACK_READELF_EXECUTABLE)
      foreach(_FILE IN LISTS CPACK_DEB_SHARED_OBJECT_FILES)
        extract_so_info("${_FILE}" libname soversion)
        if(libname AND DEFINED soversion)
          list(APPEND CPACK_DEBIAN_PACKAGE_SHLIBS_LIST
               "${libname} ${soversion} ${CPACK_DEBIAN_PACKAGE_NAME} (${CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS_POLICY} ${CPACK_DEBIAN_PACKAGE_VERSION})")
        else()
          message(AUTHOR_WARNING "Shared library '${_FILE}' is missing soname or soversion. Library will not be added to DEBIAN/shlibs control file.")
        endif()
      endforeach()
      list(JOIN CPACK_DEBIAN_PACKAGE_SHLIBS_LIST "\n" CPACK_DEBIAN_PACKAGE_SHLIBS_LIST)
    else()
      message(FATAL_ERROR "Readelf utility is not available. CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS option is not available.")
    endif()
  endif()

  # add ldconfig call in default postrm and postint
  set(CPACK_ADD_LDCONFIG_CALL 0)
  # all files in CPACK_DEB_SHARED_OBJECT_FILES have dot at the beginning
  set(_LDCONF_DEFAULTS "./lib" "./usr/lib")
  foreach(_FILE IN LISTS CPACK_DEB_SHARED_OBJECT_FILES)
    get_filename_component(_DIR ${_FILE} DIRECTORY)
    get_filename_component(_PARENT_DIR ${_DIR} DIRECTORY)
    if(_DIR IN_LIST _LDCONF_DEFAULTS OR _PARENT_DIR IN_LIST _LDCONF_DEFAULTS)
      set(CPACK_ADD_LDCONFIG_CALL 1)
    endif()
  endforeach()

  if(CPACK_ADD_LDCONFIG_CALL)
    set(CPACK_DEBIAN_GENERATE_POSTINST 1)
    set(CPACK_DEBIAN_GENERATE_POSTRM 1)
    foreach(f IN LISTS CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA)
      get_filename_component(n "${f}" NAME)
      if(n STREQUAL "postinst")
        set(CPACK_DEBIAN_GENERATE_POSTINST 0)
      endif()
      if(n STREQUAL "postrm")
        set(CPACK_DEBIAN_GENERATE_POSTRM 0)
      endif()
    endforeach()
  else()
    set(CPACK_DEBIAN_GENERATE_POSTINST 0)
    set(CPACK_DEBIAN_GENERATE_POSTRM 0)
  endif()

  cpack_deb_variable_fallback("CPACK_DEBIAN_FILE_NAME"
    "CPACK_DEBIAN_${_local_component_name}_FILE_NAME"
    "CPACK_DEBIAN_FILE_NAME")
  if(CPACK_DEBIAN_FILE_NAME)
    if(CPACK_DEBIAN_FILE_NAME STREQUAL "DEB-DEFAULT")
      # Patch package file name to be in correct debian format:
      # <foo>_<VersionNumber>-<DebianRevisionNumber>_<DebianArchitecture>.deb
      set(CPACK_OUTPUT_FILE_NAME
        "${CPACK_DEBIAN_PACKAGE_NAME}_${CPACK_DEBIAN_PACKAGE_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb")
      set(CPACK_DBGSYM_OUTPUT_FILE_NAME
        "${CPACK_DEBIAN_PACKAGE_NAME}-dbgsym_${CPACK_DEBIAN_PACKAGE_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.ddeb")
    else()
      if(NOT CPACK_DEBIAN_FILE_NAME MATCHES ".*\\.(deb|ipk)")
        set(CPACK_DEBIAN_FILE_NAME "${CPACK_DEBIAN_FILE_NAME}.deb")
      endif()

      set(CPACK_OUTPUT_FILE_NAME "${CPACK_DEBIAN_FILE_NAME}")
      string(REGEX REPLACE "\.deb$" "-dbgsym.ddeb" CPACK_DBGSYM_OUTPUT_FILE_NAME "${CPACK_DEBIAN_FILE_NAME}")
    endif()

    set(CPACK_TEMPORARY_PACKAGE_FILE_NAME "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_OUTPUT_FILE_NAME}")
    get_filename_component(BINARY_DIR "${CPACK_OUTPUT_FILE_PATH}" DIRECTORY)
    set(CPACK_OUTPUT_FILE_PATH "${BINARY_DIR}/${CPACK_OUTPUT_FILE_NAME}")
  else()
    # back compatibility - don't change the name
    string(REGEX REPLACE "\.deb$" "-dbgsym.ddeb" CPACK_DBGSYM_OUTPUT_FILE_NAME "${CPACK_OUTPUT_FILE_NAME}")
  endif()

  # Print out some debug information if we were asked for that
  if(CPACK_DEBIAN_PACKAGE_DEBUG)
     message("CPackDeb:Debug: CPACK_TOPLEVEL_DIRECTORY          = '${CPACK_TOPLEVEL_DIRECTORY}'")
     message("CPackDeb:Debug: CPACK_TOPLEVEL_TAG                = '${CPACK_TOPLEVEL_TAG}'")
     message("CPackDeb:Debug: CPACK_TEMPORARY_DIRECTORY         = '${CPACK_TEMPORARY_DIRECTORY}'")
     message("CPackDeb:Debug: CPACK_OUTPUT_FILE_NAME            = '${CPACK_OUTPUT_FILE_NAME}'")
     message("CPackDeb:Debug: CPACK_OUTPUT_FILE_PATH            = '${CPACK_OUTPUT_FILE_PATH}'")
     message("CPackDeb:Debug: CPACK_PACKAGE_FILE_NAME           = '${CPACK_PACKAGE_FILE_NAME}'")
     message("CPackDeb:Debug: CPACK_PACKAGE_INSTALL_DIRECTORY   = '${CPACK_PACKAGE_INSTALL_DIRECTORY}'")
     message("CPackDeb:Debug: CPACK_TEMPORARY_PACKAGE_FILE_NAME = '${CPACK_TEMPORARY_PACKAGE_FILE_NAME}'")
     message("CPackDeb:Debug: CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION = '${CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION}'")
     message("CPackDeb:Debug: CPACK_DEBIAN_PACKAGE_SOURCE       = '${CPACK_DEBIAN_PACKAGE_SOURCE}'")
  endif()

  # For debian source packages:
  # debian/control
  # http://www.debian.org/doc/debian-policy/ch-controlfields.html#s-sourcecontrolfiles

  # .dsc
  # http://www.debian.org/doc/debian-policy/ch-controlfields.html#s-debiansourcecontrolfiles

  # Builds-Depends:
  #if(NOT CPACK_DEBIAN_PACKAGE_BUILDS_DEPENDS)
  #  set(CPACK_DEBIAN_PACKAGE_BUILDS_DEPENDS
  #    "debhelper (>> 5.0.0), libncurses5-dev, tcl8.4"
  #  )
  #endif()

  # move variables to parent scope so that they may be used to create debian package
  set(GEN_CPACK_OUTPUT_FILE_NAME "${CPACK_OUTPUT_FILE_NAME}" PARENT_SCOPE)
  set(GEN_CPACK_TEMPORARY_PACKAGE_FILE_NAME "${CPACK_TEMPORARY_PACKAGE_FILE_NAME}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_NAME "${CPACK_DEBIAN_PACKAGE_NAME}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_VERSION "${CPACK_DEBIAN_PACKAGE_VERSION}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_SECTION "${CPACK_DEBIAN_PACKAGE_SECTION}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_PRIORITY "${CPACK_DEBIAN_PACKAGE_PRIORITY}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_ARCHITECTURE "${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_DEBIAN_PACKAGE_MAINTAINER}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_DESCRIPTION "${CPACK_DEBIAN_PACKAGE_DESCRIPTION}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_ARCHIVE_TYPE "${CPACK_DEBIAN_ARCHIVE_TYPE}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_COMPRESSION_TYPE "${CPACK_DEBIAN_COMPRESSION_TYPE}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_RECOMMENDS "${CPACK_DEBIAN_PACKAGE_RECOMMENDS}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_SUGGESTS "${CPACK_DEBIAN_PACKAGE_SUGGESTS}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_HOMEPAGE "${CPACK_DEBIAN_PACKAGE_HOMEPAGE}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_PREDEPENDS "${CPACK_DEBIAN_PACKAGE_PREDEPENDS}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_ENHANCES "${CPACK_DEBIAN_PACKAGE_ENHANCES}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_BREAKS "${CPACK_DEBIAN_PACKAGE_BREAKS}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_CONFLICTS "${CPACK_DEBIAN_PACKAGE_CONFLICTS}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_PROVIDES "${CPACK_DEBIAN_PACKAGE_PROVIDES}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_REPLACES "${CPACK_DEBIAN_PACKAGE_REPLACES}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_MULTIARCH "${CPACK_DEBIAN_PACKAGE_MULTIARCH}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_SHLIBS "${CPACK_DEBIAN_PACKAGE_SHLIBS_LIST}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION
      "${CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_PACKAGE_SOURCE
     "${CPACK_DEBIAN_PACKAGE_SOURCE}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_GENERATE_POSTINST "${CPACK_DEBIAN_GENERATE_POSTINST}" PARENT_SCOPE)
  set(GEN_CPACK_DEBIAN_GENERATE_POSTRM "${CPACK_DEBIAN_GENERATE_POSTRM}" PARENT_SCOPE)
  set(GEN_WDIR "${WDIR}" PARENT_SCOPE)

  set(GEN_CPACK_DEBIAN_DEBUGINFO_PACKAGE "${CPACK_DEBIAN_DEBUGINFO_PACKAGE}" PARENT_SCOPE)
  if(BUILD_IDS)
    set(GEN_DBGSYMDIR "${DBGSYMDIR}" PARENT_SCOPE)
    set(GEN_CPACK_DBGSYM_OUTPUT_FILE_NAME "${CPACK_DBGSYM_OUTPUT_FILE_NAME}" PARENT_SCOPE)
    list(JOIN BUILD_IDS " " BUILD_IDS)
    set(GEN_BUILD_IDS "${BUILD_IDS}" PARENT_SCOPE)
  else()
    unset(GEN_DBGSYMDIR PARENT_SCOPE)
    unset(GEN_CPACK_DBGSYM_OUTPUT_FILE_NAME PARENT_SCOPE)
    unset(GEN_BUILD_IDS PARENT_SCOPE)
  endif()
endfunction()

cpack_deb_prepare_package_vars()
