# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Author: Eric Noulard with the help of Alexander Neundorf.

function(set_spec_script_if_enabled TYPE PACKAGE_NAME VAR)
  if(NOT "${VAR}" STREQUAL "" AND NOT "${VAR}" STREQUAL "\n")
    if(PACKAGE_NAME)
      set(PACKAGE_NAME " -n ${PACKAGE_NAME}")
    endif()
    set(${TYPE}_ "%${TYPE}${PACKAGE_NAME}\n${VAR}\n" PARENT_SCOPE)
  else()
    set(${TYPE} "" PARENT_SCOPE)
  endif()
endfunction()

macro(set_spec_scripts PACKAGE_NAME)
  # we should only set scripts that were provided
  # as script announcement without content inside
  # spec file will generate unneeded dependency
  # on shell

  set_spec_script_if_enabled(
    "post"
    "${PACKAGE_NAME}"
    "${RPM_SYMLINK_POSTINSTALL}\n${CPACK_RPM_SPEC_POSTINSTALL}")

  set_spec_script_if_enabled(
    "posttrans"
    "${PACKAGE_NAME}"
    "${CPACK_RPM_SPEC_POSTTRANS}")

  set_spec_script_if_enabled(
    "postun"
    "${PACKAGE_NAME}"
    "${CPACK_RPM_SPEC_POSTUNINSTALL}")

  set_spec_script_if_enabled(
    "pre"
    "${PACKAGE_NAME}"
    "${CPACK_RPM_SPEC_PREINSTALL}")

  set_spec_script_if_enabled(
    "pretrans"
    "${PACKAGE_NAME}"
    "${CPACK_RPM_SPEC_PRETRANS}")

  set_spec_script_if_enabled(
    "preun"
    "${PACKAGE_NAME}"
    "${CPACK_RPM_SPEC_PREUNINSTALL}")
endmacro()

function(make_rpm_spec_path var path)
  # RPM supports either whitespace with quoting or globbing without quoting.
  if(path MATCHES "[ \t]")
    set("${var}" "\"${path}\"" PARENT_SCOPE)
  else()
    set("${var}" "${path}" PARENT_SCOPE)
  endif()
endfunction()

function(get_file_permissions FILE RETURN_VAR)
  execute_process(COMMAND ls -l ${FILE}
          OUTPUT_VARIABLE permissions_
          ERROR_QUIET
          OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REPLACE " " ";" permissions_ "${permissions_}")
  list(GET permissions_ 0 permissions_)

  unset(text_notation_)
  set(any_chars_ ".")
  foreach(PERMISSION_TYPE "OWNER" "GROUP" "WORLD")
    if(permissions_ MATCHES "${any_chars_}r.*")
      list(APPEND text_notation_ "${PERMISSION_TYPE}_READ")
    endif()
    string(APPEND any_chars_ ".")
    if(permissions_ MATCHES "${any_chars_}w.*")
      list(APPEND text_notation_ "${PERMISSION_TYPE}_WRITE")
    endif()
    string(APPEND any_chars_ ".")
    if(permissions_ MATCHES "${any_chars_}x.*")
      list(APPEND text_notation_ "${PERMISSION_TYPE}_EXECUTE")
    endif()
  endforeach()

  set(${RETURN_VAR} "${text_notation_}" PARENT_SCOPE)
endfunction()

function(get_unix_permissions_octal_notation PERMISSIONS_VAR RETURN_VAR)
  set(PERMISSIONS ${${PERMISSIONS_VAR}})
  list(LENGTH PERMISSIONS PERM_LEN_PRE)
  list(REMOVE_DUPLICATES PERMISSIONS)
  list(LENGTH PERMISSIONS PERM_LEN_POST)

  if(NOT ${PERM_LEN_PRE} EQUAL ${PERM_LEN_POST})
    message(FATAL_ERROR "${PERMISSIONS_VAR} contains duplicate values.")
  endif()

  foreach(PERMISSION_TYPE "OWNER" "GROUP" "WORLD")
    set(${PERMISSION_TYPE}_PERMISSIONS 0)

    foreach(PERMISSION ${PERMISSIONS})
      if("${PERMISSION}" STREQUAL "${PERMISSION_TYPE}_READ")
        math(EXPR ${PERMISSION_TYPE}_PERMISSIONS "${${PERMISSION_TYPE}_PERMISSIONS} + 4")
      elseif("${PERMISSION}" STREQUAL "${PERMISSION_TYPE}_WRITE")
        math(EXPR ${PERMISSION_TYPE}_PERMISSIONS "${${PERMISSION_TYPE}_PERMISSIONS} + 2")
      elseif("${PERMISSION}" STREQUAL "${PERMISSION_TYPE}_EXECUTE")
        math(EXPR ${PERMISSION_TYPE}_PERMISSIONS "${${PERMISSION_TYPE}_PERMISSIONS} + 1")
      elseif(PERMISSION MATCHES "${PERMISSION_TYPE}.*")
        message(FATAL_ERROR "${PERMISSIONS_VAR} contains invalid values.")
      endif()
    endforeach()
  endforeach()

  set(${RETURN_VAR} "${OWNER_PERMISSIONS}${GROUP_PERMISSIONS}${WORLD_PERMISSIONS}" PARENT_SCOPE)
endfunction()

function(cpack_rpm_exact_regex regex_var string)
  string(REGEX REPLACE "([][+.*()^])" "\\\\\\1" regex "${string}")
  set("${regex_var}" "${regex}" PARENT_SCOPE)
endfunction()

function(cpack_rpm_prepare_relocation_paths)
  # set appropriate prefix, remove possible trailing slash and convert backslashes to slashes
  if(CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_PACKAGE_PREFIX)
    file(TO_CMAKE_PATH "${CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_PACKAGE_PREFIX}" PATH_PREFIX)
  elseif(CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_PACKAGE_PREFIX)
    file(TO_CMAKE_PATH "${CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_PACKAGE_PREFIX}" PATH_PREFIX)
  else()
    file(TO_CMAKE_PATH "${CPACK_PACKAGING_INSTALL_PREFIX}" PATH_PREFIX)
  endif()

  set(RPM_RELOCATION_PATHS "${CPACK_RPM_RELOCATION_PATHS}")
  list(REMOVE_DUPLICATES RPM_RELOCATION_PATHS)

  # set base path prefix
  if(EXISTS "${WDIR}/${PATH_PREFIX}")
    if(NOT CPACK_RPM_NO_INSTALL_PREFIX_RELOCATION AND
       NOT CPACK_RPM_NO_${CPACK_RPM_PACKAGE_COMPONENT}_INSTALL_PREFIX_RELOCATION AND
       NOT CPACK_RPM_NO_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_INSTALL_PREFIX_RELOCATION)
      string(APPEND TMP_RPM_PREFIXES "Prefix: ${PATH_PREFIX}\n")
      list(APPEND RPM_USED_PACKAGE_PREFIXES "${PATH_PREFIX}")

      if(CPACK_RPM_PACKAGE_DEBUG)
        message("CPackRPM:Debug: removing '${PATH_PREFIX}' from relocation paths")
      endif()
    endif()
  endif()

  # set other path prefixes
  foreach(RELOCATION_PATH ${RPM_RELOCATION_PATHS})
    if(IS_ABSOLUTE "${RELOCATION_PATH}")
      set(PREPARED_RELOCATION_PATH "${RELOCATION_PATH}")
    elseif(PATH_PREFIX STREQUAL "/")
      # don't prefix path with a second slash as "//" is treated as network path
      # by get_filename_component() so it remains in path even inside rpm
      # package where it may cause problems with relocation
      set(PREPARED_RELOCATION_PATH "/${RELOCATION_PATH}")
    else()
      set(PREPARED_RELOCATION_PATH "${PATH_PREFIX}/${RELOCATION_PATH}")
    endif()

    # handle cases where path contains extra slashes (e.g. /a//b/ instead of
    # /a/b)
    get_filename_component(PREPARED_RELOCATION_PATH
      "${PREPARED_RELOCATION_PATH}" ABSOLUTE)

    if(EXISTS "${WDIR}/${PREPARED_RELOCATION_PATH}")
      string(APPEND TMP_RPM_PREFIXES "Prefix: ${PREPARED_RELOCATION_PATH}\n")
      list(APPEND RPM_USED_PACKAGE_PREFIXES "${PREPARED_RELOCATION_PATH}")
    endif()
  endforeach()

  # warn about all the paths that are not relocatable
  file(GLOB_RECURSE FILE_PATHS_ "${WDIR}/*")
  foreach(TMP_PATH ${FILE_PATHS_})
    string(LENGTH "${WDIR}" WDIR_LEN)
    string(SUBSTRING "${TMP_PATH}" ${WDIR_LEN} -1 TMP_PATH)
    unset(TMP_PATH_FOUND_)

    foreach(RELOCATION_PATH ${RPM_USED_PACKAGE_PREFIXES})
      file(RELATIVE_PATH REL_PATH_ "${RELOCATION_PATH}" "${TMP_PATH}")
      string(SUBSTRING "${REL_PATH_}" 0 2 PREFIX_)

      if(NOT "${PREFIX_}" STREQUAL "..")
        set(TMP_PATH_FOUND_ TRUE)
        break()
      endif()
    endforeach()

    if(NOT TMP_PATH_FOUND_)
      message(AUTHOR_WARNING "CPackRPM:Warning: Path ${TMP_PATH} is not on one of the relocatable paths! Package will be partially relocatable.")
    endif()
  endforeach()

  set(RPM_USED_PACKAGE_PREFIXES "${RPM_USED_PACKAGE_PREFIXES}" PARENT_SCOPE)
  set(TMP_RPM_PREFIXES "${TMP_RPM_PREFIXES}" PARENT_SCOPE)
endfunction()

function(cpack_rpm_prepare_content_list)
  # get files list
  file(GLOB_RECURSE CPACK_RPM_INSTALL_FILES LIST_DIRECTORIES true RELATIVE "${WDIR}" "${WDIR}/*")
  set(CPACK_RPM_INSTALL_FILES "/${CPACK_RPM_INSTALL_FILES}")
  string(REPLACE ";" ";/" CPACK_RPM_INSTALL_FILES "${CPACK_RPM_INSTALL_FILES}")

  # if we are creating a relocatable package, omit parent directories of
  # CPACK_RPM_PACKAGE_PREFIX. This is achieved by building a "filter list"
  # which is passed to the find command that generates the content-list
  if(CPACK_RPM_PACKAGE_RELOCATABLE)
    # get a list of the elements in CPACK_RPM_PACKAGE_PREFIXES that are
    # destinct parent paths of other relocation paths and remove the
    # final element (so the install-prefix dir itself is not omitted
    # from the RPM's content-list)
    list(SORT RPM_USED_PACKAGE_PREFIXES)
    set(_DISTINCT_PATH "NOT_SET")
    foreach(_RPM_RELOCATION_PREFIX ${RPM_USED_PACKAGE_PREFIXES})
      if(NOT "${_RPM_RELOCATION_PREFIX}" MATCHES "${_DISTINCT_PATH}/.*")
        set(_DISTINCT_PATH "${_RPM_RELOCATION_PREFIX}")

        string(REPLACE "/" ";" _CPACK_RPM_PACKAGE_PREFIX_ELEMS " ${_RPM_RELOCATION_PREFIX}")
        list(REMOVE_AT _CPACK_RPM_PACKAGE_PREFIX_ELEMS -1)
        unset(_TMP_LIST)
        # Now generate all of the parent dirs of the relocation path
        foreach(_PREFIX_PATH_ELEM ${_CPACK_RPM_PACKAGE_PREFIX_ELEMS})
          list(APPEND _TMP_LIST "${_PREFIX_PATH_ELEM}")
          string(REPLACE ";" "/" _OMIT_DIR "${_TMP_LIST}")
          separate_arguments(_OMIT_DIR)
          list(APPEND _RPM_DIRS_TO_OMIT ${_OMIT_DIR})
        endforeach()
      endif()
    endforeach()
  endif()

  if(CPACK_RPM_PACKAGE_DEBUG)
    message("CPackRPM:Debug: Initial list of path to OMIT in RPM: ${_RPM_DIRS_TO_OMIT}")
  endif()

  if(NOT DEFINED CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST)
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST /etc /etc/init.d /usr /usr/bin
        /usr/include /usr/lib /usr/libx32 /usr/lib64
        /usr/share /usr/share/aclocal /usr/share/doc )
    if(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION)
      if(CPACK_RPM_PACKAGE_DEBUG)
        message("CPackRPM:Debug: Adding ${CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION} to builtin omit list.")
      endif()
      list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST "${CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION}")
    endif()
  endif()

  if(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST)
    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST= ${CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST}")
    endif()
    list(APPEND _RPM_DIRS_TO_OMIT ${CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST})
  endif()

  if(CPACK_RPM_PACKAGE_DEBUG)
    message("CPackRPM:Debug: Final list of path to OMIT in RPM: ${_RPM_DIRS_TO_OMIT}")
  endif()

  list(REMOVE_ITEM CPACK_RPM_INSTALL_FILES ${_RPM_DIRS_TO_OMIT})

  # add man paths that will be compressed
  # (copied from /usr/lib/rpm/brp-compress - script that does the actual
  # compressing)
  list(APPEND MAN_LOCATIONS "/usr/man/man.*" "/usr/man/.*/man.*" "/usr/info.*"
    "/usr/share/man/man.*" "/usr/share/man/.*/man.*" "/usr/share/info.*"
    "/usr/kerberos/man.*" "/usr/X11R6/man/man.*" "/usr/lib/perl5/man/man.*"
    "/usr/share/doc/.*/man/man.*" "/usr/lib/.*/man/man.*")

  if(CPACK_RPM_ADDITIONAL_MAN_DIRS)
    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: CPACK_RPM_ADDITIONAL_MAN_DIRS= ${CPACK_RPM_ADDITIONAL_MAN_DIRS}")
    endif()
    list(APPEND MAN_LOCATIONS ${CPACK_RPM_ADDITIONAL_MAN_DIRS})
  endif()

  foreach(PACK_LOCATION IN LISTS CPACK_RPM_INSTALL_FILES)
    foreach(MAN_LOCATION IN LISTS MAN_LOCATIONS)
      # man pages are files inside a certain location
      if(PACK_LOCATION MATCHES "${MAN_LOCATION}/"
        AND NOT IS_DIRECTORY "${WDIR}${PACK_LOCATION}"
        AND NOT IS_SYMLINK "${WDIR}${PACK_LOCATION}")
        list(FIND CPACK_RPM_INSTALL_FILES "${PACK_LOCATION}" INDEX)
        # insert file location that covers compressed man pages
        # even if using a wildcard causes duplicates as those are
        # handled by RPM and we still keep the same file list
        # in spec file - wildcard only represents file type (e.g. .gz)
        list(INSERT CPACK_RPM_INSTALL_FILES ${INDEX} "${PACK_LOCATION}*")
        # remove file location that doesn't cover compressed man pages
        math(EXPR INDEX ${INDEX}+1)
        list(REMOVE_AT CPACK_RPM_INSTALL_FILES ${INDEX})

        break()
      endif()
    endforeach()
  endforeach()

  set(CPACK_RPM_INSTALL_FILES "${CPACK_RPM_INSTALL_FILES}" PARENT_SCOPE)
endfunction()

function(cpack_rpm_symlink_get_relocation_prefixes LOCATION PACKAGE_PREFIXES RETURN_VARIABLE)
  foreach(PKG_PREFIX IN LISTS PACKAGE_PREFIXES)
    string(REGEX MATCH "^${PKG_PREFIX}/.*" FOUND_ "${LOCATION}")
    if(FOUND_)
      list(APPEND TMP_PREFIXES "${PKG_PREFIX}")
    endif()
  endforeach()

  set(${RETURN_VARIABLE} "${TMP_PREFIXES}" PARENT_SCOPE)
endfunction()

function(cpack_rpm_symlink_create_relocation_script PACKAGE_PREFIXES)
  list(LENGTH PACKAGE_PREFIXES LAST_INDEX)
  set(SORTED_PACKAGE_PREFIXES "${PACKAGE_PREFIXES}")
  list(SORT SORTED_PACKAGE_PREFIXES)
  list(REVERSE SORTED_PACKAGE_PREFIXES)
  math(EXPR LAST_INDEX ${LAST_INDEX}-1)

  foreach(SYMLINK_INDEX RANGE ${LAST_INDEX})
    list(GET SORTED_PACKAGE_PREFIXES ${SYMLINK_INDEX} SRC_PATH)
    list(FIND PACKAGE_PREFIXES "${SRC_PATH}" SYMLINK_INDEX) # reverse magic
    string(LENGTH "${SRC_PATH}" SRC_PATH_LEN)

    set(PARTS_CNT 0)
    set(SCRIPT_PART "if [ \"$RPM_INSTALL_PREFIX${SYMLINK_INDEX}\" != \"${SRC_PATH}\" ]; then\n")

    # both paths relocated
    foreach(POINT_INDEX RANGE ${LAST_INDEX})
      list(GET SORTED_PACKAGE_PREFIXES ${POINT_INDEX} POINT_PATH)
      list(FIND PACKAGE_PREFIXES "${POINT_PATH}" POINT_INDEX) # reverse magic
      string(LENGTH "${POINT_PATH}" POINT_PATH_LEN)

      if(_RPM_RELOCATION_SCRIPT_${SYMLINK_INDEX}_${POINT_INDEX})
        if("${SYMLINK_INDEX}" EQUAL "${POINT_INDEX}")
          set(INDENT "")
        else()
          string(APPEND SCRIPT_PART "  if [ \"$RPM_INSTALL_PREFIX${POINT_INDEX}\" != \"${POINT_PATH}\" ]; then\n")
          set(INDENT "  ")
        endif()

        foreach(RELOCATION_NO IN LISTS _RPM_RELOCATION_SCRIPT_${SYMLINK_INDEX}_${POINT_INDEX})
          math(EXPR PARTS_CNT ${PARTS_CNT}+1)

          math(EXPR RELOCATION_INDEX ${RELOCATION_NO}-1)
          list(GET _RPM_RELOCATION_SCRIPT_PAIRS ${RELOCATION_INDEX} RELOCATION_SCRIPT_PAIR)
          string(FIND "${RELOCATION_SCRIPT_PAIR}" ":" SPLIT_INDEX)

          math(EXPR SRC_PATH_END ${SPLIT_INDEX}-${SRC_PATH_LEN})
          string(SUBSTRING ${RELOCATION_SCRIPT_PAIR} ${SRC_PATH_LEN} ${SRC_PATH_END} SYMLINK_)

          math(EXPR POINT_PATH_START ${SPLIT_INDEX}+1+${POINT_PATH_LEN})
          string(SUBSTRING ${RELOCATION_SCRIPT_PAIR} ${POINT_PATH_START} -1 POINT_)

          string(APPEND SCRIPT_PART "  ${INDENT}if [ -z \"$CPACK_RPM_RELOCATED_SYMLINK_${RELOCATION_INDEX}\" ]; then\n")
          string(APPEND SCRIPT_PART "    ${INDENT}ln -s \"$RPM_INSTALL_PREFIX${POINT_INDEX}${POINT_}\" \"$RPM_INSTALL_PREFIX${SYMLINK_INDEX}${SYMLINK_}\"\n")
          string(APPEND SCRIPT_PART "    ${INDENT}CPACK_RPM_RELOCATED_SYMLINK_${RELOCATION_INDEX}=true\n")
          string(APPEND SCRIPT_PART "  ${INDENT}fi\n")
        endforeach()

        if(NOT "${SYMLINK_INDEX}" EQUAL "${POINT_INDEX}")
          string(APPEND SCRIPT_PART "  fi\n")
        endif()
      endif()
    endforeach()

    # source path relocated
    if(_RPM_RELOCATION_SCRIPT_${SYMLINK_INDEX}_X)
      foreach(RELOCATION_NO IN LISTS _RPM_RELOCATION_SCRIPT_${SYMLINK_INDEX}_X)
        math(EXPR PARTS_CNT ${PARTS_CNT}+1)

        math(EXPR RELOCATION_INDEX ${RELOCATION_NO}-1)
        list(GET _RPM_RELOCATION_SCRIPT_PAIRS ${RELOCATION_INDEX} RELOCATION_SCRIPT_PAIR)
        string(FIND "${RELOCATION_SCRIPT_PAIR}" ":" SPLIT_INDEX)

        math(EXPR SRC_PATH_END ${SPLIT_INDEX}-${SRC_PATH_LEN})
        string(SUBSTRING ${RELOCATION_SCRIPT_PAIR} ${SRC_PATH_LEN} ${SRC_PATH_END} SYMLINK_)

        math(EXPR POINT_PATH_START ${SPLIT_INDEX}+1)
        string(SUBSTRING ${RELOCATION_SCRIPT_PAIR} ${POINT_PATH_START} -1 POINT_)

        string(APPEND SCRIPT_PART "  if [ -z \"$CPACK_RPM_RELOCATED_SYMLINK_${RELOCATION_INDEX}\" ]; then\n")
        string(APPEND SCRIPT_PART "    ln -s \"${POINT_}\" \"$RPM_INSTALL_PREFIX${SYMLINK_INDEX}${SYMLINK_}\"\n")
        string(APPEND SCRIPT_PART "    CPACK_RPM_RELOCATED_SYMLINK_${RELOCATION_INDEX}=true\n")
        string(APPEND SCRIPT_PART "  fi\n")
      endforeach()
    endif()

    if(PARTS_CNT)
      set(SCRIPT "${SCRIPT_PART}")
      string(APPEND SCRIPT "fi\n")
    endif()
  endforeach()

  # point path relocated
  foreach(POINT_INDEX RANGE ${LAST_INDEX})
    list(GET SORTED_PACKAGE_PREFIXES ${POINT_INDEX} POINT_PATH)
    list(FIND PACKAGE_PREFIXES "${POINT_PATH}" POINT_INDEX) # reverse magic
    string(LENGTH "${POINT_PATH}" POINT_PATH_LEN)

    if(_RPM_RELOCATION_SCRIPT_X_${POINT_INDEX})
      string(APPEND SCRIPT "if [ \"$RPM_INSTALL_PREFIX${POINT_INDEX}\" != \"${POINT_PATH}\" ]; then\n")

      foreach(RELOCATION_NO IN LISTS _RPM_RELOCATION_SCRIPT_X_${POINT_INDEX})
        math(EXPR RELOCATION_INDEX ${RELOCATION_NO}-1)
        list(GET _RPM_RELOCATION_SCRIPT_PAIRS ${RELOCATION_INDEX} RELOCATION_SCRIPT_PAIR)
        string(FIND "${RELOCATION_SCRIPT_PAIR}" ":" SPLIT_INDEX)

        string(SUBSTRING ${RELOCATION_SCRIPT_PAIR} 0 ${SPLIT_INDEX} SYMLINK_)

        math(EXPR POINT_PATH_START ${SPLIT_INDEX}+1+${POINT_PATH_LEN})
        string(SUBSTRING ${RELOCATION_SCRIPT_PAIR} ${POINT_PATH_START} -1 POINT_)

        string(APPEND SCRIPT "  if [ -z \"$CPACK_RPM_RELOCATED_SYMLINK_${RELOCATION_INDEX}\" ]; then\n")
        string(APPEND SCRIPT "    ln -s \"$RPM_INSTALL_PREFIX${POINT_INDEX}${POINT_}\" \"${SYMLINK_}\"\n")
        string(APPEND SCRIPT "    CPACK_RPM_RELOCATED_SYMLINK_${RELOCATION_INDEX}=true\n")
        string(APPEND SCRIPT "  fi\n")
      endforeach()

      string(APPEND SCRIPT "fi\n")
    endif()
  endforeach()

  # no path relocated
  if(_RPM_RELOCATION_SCRIPT_X_X)
    foreach(RELOCATION_NO IN LISTS _RPM_RELOCATION_SCRIPT_X_X)
      math(EXPR RELOCATION_INDEX ${RELOCATION_NO}-1)
      list(GET _RPM_RELOCATION_SCRIPT_PAIRS ${RELOCATION_INDEX} RELOCATION_SCRIPT_PAIR)
      string(FIND "${RELOCATION_SCRIPT_PAIR}" ":" SPLIT_INDEX)

      string(SUBSTRING ${RELOCATION_SCRIPT_PAIR} 0 ${SPLIT_INDEX} SYMLINK_)

      math(EXPR POINT_PATH_START ${SPLIT_INDEX}+1)
      string(SUBSTRING ${RELOCATION_SCRIPT_PAIR} ${POINT_PATH_START} -1 POINT_)

      string(APPEND SCRIPT "if [ -z \"$CPACK_RPM_RELOCATED_SYMLINK_${RELOCATION_INDEX}\" ]; then\n")
      string(APPEND SCRIPT "  ln -s \"${POINT_}\" \"${SYMLINK_}\"\n")
      string(APPEND SCRIPT "fi\n")
    endforeach()
  endif()

  set(RPM_SYMLINK_POSTINSTALL "${SCRIPT}" PARENT_SCOPE)
endfunction()

function(cpack_rpm_symlink_add_for_relocation_script PACKAGE_PREFIXES SYMLINK SYMLINK_RELOCATION_PATHS POINT POINT_RELOCATION_PATHS)
  list(LENGTH SYMLINK_RELOCATION_PATHS SYMLINK_PATHS_COUNT)
  list(LENGTH POINT_RELOCATION_PATHS POINT_PATHS_COUNT)

  list(APPEND _RPM_RELOCATION_SCRIPT_PAIRS "${SYMLINK}:${POINT}")
  list(LENGTH _RPM_RELOCATION_SCRIPT_PAIRS PAIR_NO)

  if(SYMLINK_PATHS_COUNT)
    foreach(SYMLINK_RELOC_PATH IN LISTS SYMLINK_RELOCATION_PATHS)
      list(FIND PACKAGE_PREFIXES "${SYMLINK_RELOC_PATH}" SYMLINK_INDEX)

      # source path relocated
      list(APPEND _RPM_RELOCATION_SCRIPT_${SYMLINK_INDEX}_X "${PAIR_NO}")
      list(APPEND RELOCATION_VARS "_RPM_RELOCATION_SCRIPT_${SYMLINK_INDEX}_X")

      foreach(POINT_RELOC_PATH IN LISTS POINT_RELOCATION_PATHS)
        list(FIND PACKAGE_PREFIXES "${POINT_RELOC_PATH}" POINT_INDEX)

        # both paths relocated
        list(APPEND _RPM_RELOCATION_SCRIPT_${SYMLINK_INDEX}_${POINT_INDEX} "${PAIR_NO}")
        list(APPEND RELOCATION_VARS "_RPM_RELOCATION_SCRIPT_${SYMLINK_INDEX}_${POINT_INDEX}")

        # point path relocated
        list(APPEND _RPM_RELOCATION_SCRIPT_X_${POINT_INDEX} "${PAIR_NO}")
        list(APPEND RELOCATION_VARS "_RPM_RELOCATION_SCRIPT_X_${POINT_INDEX}")
      endforeach()
    endforeach()
  elseif(POINT_PATHS_COUNT)
    foreach(POINT_RELOC_PATH IN LISTS POINT_RELOCATION_PATHS)
      list(FIND PACKAGE_PREFIXES "${POINT_RELOC_PATH}" POINT_INDEX)

      # point path relocated
      list(APPEND _RPM_RELOCATION_SCRIPT_X_${POINT_INDEX} "${PAIR_NO}")
      list(APPEND RELOCATION_VARS "_RPM_RELOCATION_SCRIPT_X_${POINT_INDEX}")
    endforeach()
  endif()

  # no path relocated
  list(APPEND _RPM_RELOCATION_SCRIPT_X_X "${PAIR_NO}")
  list(APPEND RELOCATION_VARS "_RPM_RELOCATION_SCRIPT_X_X")

  # place variables into parent scope
  foreach(VAR IN LISTS RELOCATION_VARS)
    set(${VAR} "${${VAR}}" PARENT_SCOPE)
  endforeach()
  set(_RPM_RELOCATION_SCRIPT_PAIRS "${_RPM_RELOCATION_SCRIPT_PAIRS}" PARENT_SCOPE)
  set(REQUIRES_SYMLINK_RELOCATION_SCRIPT "true" PARENT_SCOPE)
  set(DIRECTIVE "%ghost " PARENT_SCOPE)
endfunction()

function(cpack_rpm_prepare_install_files INSTALL_FILES_LIST WDIR PACKAGE_PREFIXES IS_RELOCATABLE)
  # Prepend directories in ${CPACK_RPM_INSTALL_FILES} with %dir
  # This is necessary to avoid duplicate files since rpmbuild does
  # recursion on its own when encountering a pathname which is a directory
  # which is not flagged as %dir
  string(STRIP "${INSTALL_FILES_LIST}" INSTALL_FILES_LIST)
  string(REPLACE "\n" ";" INSTALL_FILES_LIST
                          "${INSTALL_FILES_LIST}")
  string(REPLACE "\"" "" INSTALL_FILES_LIST
                          "${INSTALL_FILES_LIST}")
  string(LENGTH "${WDIR}" WDR_LEN_)

  list(SORT INSTALL_FILES_LIST) # make file order consistent on all platforms

  foreach(F IN LISTS INSTALL_FILES_LIST)
    unset(DIRECTIVE)

    if(IS_SYMLINK "${WDIR}/${F}")
      if(IS_RELOCATABLE)
        # check that symlink has relocatable format
        get_filename_component(SYMLINK_LOCATION_ "${WDIR}/${F}" DIRECTORY)
        execute_process(COMMAND ls -la "${WDIR}/${F}"
                  WORKING_DIRECTORY "${WDIR}"
                  OUTPUT_VARIABLE SYMLINK_POINT_
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

        string(FIND "${SYMLINK_POINT_}" "->" SYMLINK_POINT_INDEX_ REVERSE)
        math(EXPR SYMLINK_POINT_INDEX_ ${SYMLINK_POINT_INDEX_}+3)
        string(LENGTH "${SYMLINK_POINT_}" SYMLINK_POINT_LENGTH_)

        # get destination path
        string(SUBSTRING "${SYMLINK_POINT_}" ${SYMLINK_POINT_INDEX_} ${SYMLINK_POINT_LENGTH_} SYMLINK_POINT_)

        # check if path is relative or absolute
        string(SUBSTRING "${SYMLINK_POINT_}" 0 1 SYMLINK_IS_ABSOLUTE_)

        if(${SYMLINK_IS_ABSOLUTE_} STREQUAL "/")
          # prevent absolute paths from having /../ or /./ section inside of them
          get_filename_component(SYMLINK_POINT_ "${SYMLINK_POINT_}" ABSOLUTE)
        else()
          # handle relative path
          get_filename_component(SYMLINK_POINT_ "${SYMLINK_LOCATION_}/${SYMLINK_POINT_}" ABSOLUTE)
        endif()

        # recalculate path length after conversion to canonical form
        string(LENGTH "${SYMLINK_POINT_}" SYMLINK_POINT_LENGTH_)

        cpack_rpm_exact_regex(IN_SYMLINK_POINT_REGEX "${WDIR}")
        string(APPEND IN_SYMLINK_POINT_REGEX "/.*")
        if(SYMLINK_POINT_ MATCHES "${IN_SYMLINK_POINT_REGEX}")
          # only symlinks that are pointing inside the packaging structure should be checked for relocation
          string(SUBSTRING "${SYMLINK_POINT_}" ${WDR_LEN_} -1 SYMLINK_POINT_WD_)
          cpack_rpm_symlink_get_relocation_prefixes("${F}" "${PACKAGE_PREFIXES}" "SYMLINK_RELOCATIONS")
          cpack_rpm_symlink_get_relocation_prefixes("${SYMLINK_POINT_WD_}" "${PACKAGE_PREFIXES}" "POINT_RELOCATIONS")

          list(LENGTH SYMLINK_RELOCATIONS SYMLINK_RELOCATIONS_COUNT)
          list(LENGTH POINT_RELOCATIONS POINT_RELOCATIONS_COUNT)
        else()
          # location pointed to is outside WDR so it should be treated as a permanent symlink
          set(SYMLINK_POINT_WD_ "${SYMLINK_POINT_}")

          unset(SYMLINK_RELOCATIONS)
          unset(POINT_RELOCATIONS)
          unset(SYMLINK_RELOCATIONS_COUNT)
          unset(POINT_RELOCATIONS_COUNT)

          message(AUTHOR_WARNING "CPackRPM:Warning: Symbolic link '${F}' points to location that is outside packaging path! Link will possibly not be relocatable.")
        endif()

        if(SYMLINK_RELOCATIONS_COUNT AND POINT_RELOCATIONS_COUNT)
          # find matching
          foreach(SYMLINK_RELOCATION_PREFIX IN LISTS SYMLINK_RELOCATIONS)
            list(FIND POINT_RELOCATIONS "${SYMLINK_RELOCATION_PREFIX}" FOUND_INDEX)
            if(NOT ${FOUND_INDEX} EQUAL -1)
              break()
            endif()
          endforeach()

          if(NOT ${FOUND_INDEX} EQUAL -1)
            # symlinks have the same subpath
            if(${SYMLINK_RELOCATIONS_COUNT} EQUAL 1 AND ${POINT_RELOCATIONS_COUNT} EQUAL 1)
              # permanent symlink
              get_filename_component(SYMLINK_LOCATION_ "${F}" DIRECTORY)
              file(RELATIVE_PATH FINAL_PATH_ ${SYMLINK_LOCATION_} ${SYMLINK_POINT_WD_})
              execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink "${FINAL_PATH_}" "${WDIR}/${F}")
            else()
              # relocation subpaths
              cpack_rpm_symlink_add_for_relocation_script("${PACKAGE_PREFIXES}" "${F}" "${SYMLINK_RELOCATIONS}"
                  "${SYMLINK_POINT_WD_}" "${POINT_RELOCATIONS}")
            endif()
          else()
            # not on the same relocation path
            cpack_rpm_symlink_add_for_relocation_script("${PACKAGE_PREFIXES}" "${F}" "${SYMLINK_RELOCATIONS}"
                "${SYMLINK_POINT_WD_}" "${POINT_RELOCATIONS}")
          endif()
        elseif(POINT_RELOCATIONS_COUNT)
          # point is relocatable
          cpack_rpm_symlink_add_for_relocation_script("${PACKAGE_PREFIXES}" "${F}" "${SYMLINK_RELOCATIONS}"
              "${SYMLINK_POINT_WD_}" "${POINT_RELOCATIONS}")
        else()
          # is not relocatable or points to non relocatable path - permanent symlink
          execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink "${SYMLINK_POINT_WD_}" "${WDIR}/${F}")
        endif()
      endif()
    elseif(IS_DIRECTORY "${WDIR}/${F}")
      set(DIRECTIVE "%dir ")
    endif()

    make_rpm_spec_path(F_SPEC "${F}")
    string(APPEND INSTALL_FILES "${DIRECTIVE}${F_SPEC}\n")
  endforeach()

  if(REQUIRES_SYMLINK_RELOCATION_SCRIPT)
    cpack_rpm_symlink_create_relocation_script("${PACKAGE_PREFIXES}")
  endif()

  set(RPM_SYMLINK_POSTINSTALL "${RPM_SYMLINK_POSTINSTALL}" PARENT_SCOPE)
  set(CPACK_RPM_INSTALL_FILES "${INSTALL_FILES}" PARENT_SCOPE)
endfunction()

if(CMAKE_BINARY_DIR)
  message(FATAL_ERROR "CPackRPM.cmake may only be used by CPack internally.")
endif()

if(NOT UNIX)
  message(FATAL_ERROR "CPackRPM.cmake may only be used under UNIX.")
endif()

# We need to check if the binaries were compiled with debug symbols
# because without them the package will be useless
function(cpack_rpm_debugsymbol_check INSTALL_FILES WORKING_DIR)
  if(NOT CPACK_BUILD_SOURCE_DIRS)
    message(FATAL_ERROR "CPackRPM: CPACK_BUILD_SOURCE_DIRS variable is not set!"
      " Required for debuginfo packaging. See documentation of"
      " CPACK_RPM_DEBUGINFO_PACKAGE variable for details.")
  endif()

  # With objdump we should check the debug symbols
  find_program(CPACK_OBJDUMP_EXECUTABLE objdump)
  if(NOT CPACK_OBJDUMP_EXECUTABLE)
    message(FATAL_ERROR "CPackRPM: objdump binary could not be found!"
      " Required for debuginfo packaging. See documentation of"
      " CPACK_RPM_DEBUGINFO_PACKAGE variable for details.")
  endif()

  # With debugedit we prepare source files list
  find_program(DEBUGEDIT_EXECUTABLE debugedit "/usr/lib/rpm/")
  if(NOT DEBUGEDIT_EXECUTABLE)
    message(FATAL_ERROR "CPackRPM: debugedit binary could not be found!"
      " Required for debuginfo packaging. See documentation of"
      " CPACK_RPM_DEBUGINFO_PACKAGE variable for details.")
  endif()

  unset(mkdir_list_)
  unset(cp_list_)
  unset(additional_sources_)

  foreach(F IN LISTS INSTALL_FILES)
    if(IS_DIRECTORY "${WORKING_DIR}/${F}" OR IS_SYMLINK "${WORKING_DIR}/${F}")
      continue()
    endif()

    execute_process(COMMAND "${CPACK_OBJDUMP_EXECUTABLE}" -h ${WORKING_DIR}/${F}
                    WORKING_DIRECTORY "${CPACK_TOPLEVEL_DIRECTORY}"
                    RESULT_VARIABLE OBJDUMP_EXEC_RESULT
                    OUTPUT_VARIABLE OBJDUMP_OUT
                    ERROR_QUIET)
    # Check if the given file is an executable or not
    if(NOT OBJDUMP_EXEC_RESULT)
      string(FIND "${OBJDUMP_OUT}" "debug" FIND_RESULT)
      if(FIND_RESULT GREATER -1)
        set(index_ 0)
        foreach(source_dir_ IN LISTS CPACK_BUILD_SOURCE_DIRS)
          string(LENGTH "${source_dir_}" source_dir_len_)
          string(LENGTH "${CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX}/src_${index_}" debuginfo_dir_len)
          if(source_dir_len_ LESS debuginfo_dir_len)
            message(FATAL_ERROR "CPackRPM: source dir path '${source_dir_}' is"
              " shorter than debuginfo sources dir path '${CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX}/src_${index_}'!"
              " Source dir path must be longer than debuginfo sources dir path."
              " Set CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX variable to a shorter value"
              " or make source dir path longer."
              " Required for debuginfo packaging. See documentation of"
              " CPACK_RPM_DEBUGINFO_PACKAGE variable for details.")
          endif()

          file(REMOVE "${CPACK_RPM_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}${CPACK_RPM_PACKAGE_COMPONENT_PART_PATH}/debugsources_add.list")
          execute_process(COMMAND "${DEBUGEDIT_EXECUTABLE}" -b "${source_dir_}" -d "${CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX}/src_${index_}" -i -l "${CPACK_RPM_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}${CPACK_RPM_PACKAGE_COMPONENT_PART_PATH}/debugsources_add.list" "${WORKING_DIR}/${F}"
              RESULT_VARIABLE res_
              OUTPUT_VARIABLE opt_
              ERROR_VARIABLE err_
            )

          file(STRINGS
            "${CPACK_RPM_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}${CPACK_RPM_PACKAGE_COMPONENT_PART_PATH}/debugsources_add.list"
            sources_)
          list(REMOVE_DUPLICATES sources_)

          foreach(source_ IN LISTS sources_)
            if(EXISTS "${source_dir_}/${source_}" AND NOT IS_DIRECTORY "${source_dir_}/${source_}")
              get_filename_component(path_part_ "${source_}" DIRECTORY)
              list(APPEND mkdir_list_ "%{buildroot}${CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX}/src_${index_}/${path_part_}")
              list(APPEND cp_list_ "cp \"${source_dir_}/${source_}\" \"%{buildroot}${CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX}/src_${index_}/${path_part_}\"")

              list(APPEND additional_sources_ "${CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX}/src_${index_}/${source_}")
            endif()
          endforeach()

          math(EXPR index_ "${index_} + 1")
        endforeach()
      else()
        message(WARNING "CPackRPM: File: ${F} does not contain debug symbols. They will possibly be missing from debuginfo package!")
      endif()

      get_file_permissions("${WORKING_DIR}/${F}" permissions_)
      if(NOT "OWNER_EXECUTE" IN_LIST permissions_ AND
         NOT "GROUP_EXECUTE" IN_LIST permissions_ AND
         NOT "WORLD_EXECUTE" IN_LIST permissions_)
        if(CPACK_RPM_INSTALL_WITH_EXEC)
          execute_process(COMMAND chmod a+x ${WORKING_DIR}/${F}
                  RESULT_VARIABLE res_
                  ERROR_VARIABLE err_
                  OUTPUT_QUIET)

          if(res_)
            message(FATAL_ERROR "CPackRPM: could not apply execute permissions "
              "requested by CPACK_RPM_INSTALL_WITH_EXEC variable on "
              "'${WORKING_DIR}/${F}'! Reason: '${err_}'")
          endif()
        else()
          message(AUTHOR_WARNING "CPackRPM: File: ${WORKING_DIR}/${F} does not "
            "have execute permissions. Debuginfo symbols will not be extracted"
            "! Missing debuginfo may cause packaging failure. Consider setting "
            "execute permissions or setting 'CPACK_RPM_INSTALL_WITH_EXEC' "
            "variable.")
        endif()
      endif()
    endif()
  endforeach()

  list(LENGTH mkdir_list_ len_)
  if(len_)
    list(REMOVE_DUPLICATES mkdir_list_)
    unset(TMP_RPM_DEBUGINFO_INSTALL)
    foreach(part_ IN LISTS mkdir_list_)
      string(APPEND TMP_RPM_DEBUGINFO_INSTALL "mkdir -p \"${part_}\"\n")
    endforeach()
  endif()

  list(LENGTH cp_list_ len_)
  if(len_)
    list(REMOVE_DUPLICATES cp_list_)
    foreach(part_ IN LISTS cp_list_)
      string(APPEND TMP_RPM_DEBUGINFO_INSTALL "${part_}\n")
    endforeach()
  endif()

  if(NOT DEFINED CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS)
    set(CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS /usr /usr/src /usr/src/debug)
    if(CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS_ADDITION)
      if(CPACK_RPM_PACKAGE_DEBUG)
        message("CPackRPM:Debug: Adding ${CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS_ADDITION} to builtin omit list.")
      endif()
      list(APPEND CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS "${CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS_ADDITION}")
    endif()
  endif()
  if(CPACK_RPM_PACKAGE_DEBUG)
    message("CPackRPM:Debug: CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS= ${CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS}")
  endif()

  list(LENGTH additional_sources_ len_)
  if(len_)
    list(REMOVE_DUPLICATES additional_sources_)
    unset(additional_sources_all_)
    foreach(source_ IN LISTS additional_sources_)
      string(REPLACE "/" ";" split_source_ " ${source_}")
      list(REMOVE_AT split_source_ 0)
      unset(tmp_path_)
      # Now generate all segments of the path
      foreach(segment_ IN LISTS split_source_)
        string(APPEND tmp_path_ "/${segment_}")
        list(APPEND additional_sources_all_ "${tmp_path_}")
      endforeach()
    endforeach()

    list(REMOVE_DUPLICATES additional_sources_all_)
    list(REMOVE_ITEM additional_sources_all_
      ${CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS})

    unset(TMP_DEBUGINFO_ADDITIONAL_SOURCES)
    foreach(source_ IN LISTS additional_sources_all_)
      string(APPEND TMP_DEBUGINFO_ADDITIONAL_SOURCES "${source_}\n")
    endforeach()
  endif()

  set(TMP_RPM_DEBUGINFO_INSTALL "${TMP_RPM_DEBUGINFO_INSTALL}" PARENT_SCOPE)
  set(TMP_DEBUGINFO_ADDITIONAL_SOURCES "${TMP_DEBUGINFO_ADDITIONAL_SOURCES}"
    PARENT_SCOPE)
endfunction()

function(cpack_rpm_variable_fallback OUTPUT_VAR_NAME)
  set(FALLBACK_VAR_NAMES ${ARGN})

  foreach(variable_name IN LISTS FALLBACK_VAR_NAMES)
    if(DEFINED ${variable_name})
      set(${OUTPUT_VAR_NAME} "${${variable_name}}" PARENT_SCOPE)
      break()
    endif()
  endforeach()
endfunction()

function(cpack_rpm_generate_package)
  # rpmbuild is the basic command for building RPM package
  # it may be a simple (symbolic) link to rpm command.
  find_program(RPMBUILD_EXECUTABLE rpmbuild)
  if(NOT RPMBUILD_EXECUTABLE)
    message(FATAL_ERROR "RPM package requires rpmbuild executable")
  endif()

  # rpm is used for fallback queries in some older versions,
  # but is not required in general.
  find_program(RPM_EXECUTABLE rpm)

  # Check version of the rpmbuild tool this would be easier to
  # track bugs with users and CPackRPM debug mode.
  # We may use RPM version in order to check for available version dependent features
  execute_process(COMMAND ${RPMBUILD_EXECUTABLE} --version
                  OUTPUT_VARIABLE _TMP_VERSION
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX REPLACE "^.* " ""
         RPMBUILD_EXECUTABLE_VERSION
         ${_TMP_VERSION})
  if(CPACK_RPM_PACKAGE_DEBUG)
    message("CPackRPM:Debug: rpmbuild version is <${RPMBUILD_EXECUTABLE_VERSION}>")
  endif()

  # Display lsb_release output if DEBUG mode enable
  # This will help to diagnose problem with CPackRPM
  # because we will know on which kind of Linux we are
  if(CPACK_RPM_PACKAGE_DEBUG)
    find_program(LSB_RELEASE_EXECUTABLE lsb_release)
    if(LSB_RELEASE_EXECUTABLE)
      execute_process(COMMAND ${LSB_RELEASE_EXECUTABLE} -a
                      OUTPUT_VARIABLE _TMP_LSB_RELEASE_OUTPUT
                      ERROR_QUIET
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REGEX REPLACE "\n" ", "
             LSB_RELEASE_OUTPUT
             ${_TMP_LSB_RELEASE_OUTPUT})
    else ()
      set(LSB_RELEASE_OUTPUT "lsb_release not installed/found!")
    endif()
    message("CPackRPM:Debug: LSB_RELEASE  = ${LSB_RELEASE_OUTPUT}")
  endif()

  # We may use RPM version in the future in order
  # to shut down warning about space in buildtree
  # some recent RPM version should support space in different places.
  # not checked [yet].
  if(CPACK_TOPLEVEL_DIRECTORY MATCHES ".* .*")
    message(FATAL_ERROR "${RPMBUILD_EXECUTABLE} can't handle paths with spaces, use a build directory without spaces for building RPMs.")
  endif()

  # Are we packaging components ?
  if(CPACK_RPM_PACKAGE_COMPONENT)
    string(TOUPPER ${CPACK_RPM_PACKAGE_COMPONENT} CPACK_RPM_PACKAGE_COMPONENT_UPPER)
  endif()

  set(WDIR "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}${CPACK_RPM_PACKAGE_COMPONENT_PART_PATH}")

  #
  # Use user-defined RPM specific variables value
  # or generate reasonable default value from
  # CPACK_xxx generic values.
  # The variables comes from the needed (mandatory or not)
  # values found in the RPM specification file aka ".spec" file.
  # The variables which may/should be defined are:
  #

  # CPACK_RPM_PACKAGE_SUMMARY (mandatory)

  if(CPACK_RPM_PACKAGE_COMPONENT)
    cpack_rpm_variable_fallback("CPACK_RPM_PACKAGE_SUMMARY"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_PACKAGE_SUMMARY"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_PACKAGE_SUMMARY")
  endif()

  if(NOT CPACK_RPM_PACKAGE_SUMMARY)
    if(CPACK_PACKAGE_DESCRIPTION_SUMMARY)
      set(CPACK_RPM_PACKAGE_SUMMARY ${CPACK_PACKAGE_DESCRIPTION_SUMMARY})
    else()
      # if neither var is defined lets use the name as summary
      string(TOLOWER "${CPACK_PACKAGE_NAME}" CPACK_RPM_PACKAGE_SUMMARY)
    endif()
  endif()

  if(NOT CPACK_RPM_PACKAGE_URL AND CPACK_PACKAGE_HOMEPAGE_URL)
    set(CPACK_RPM_PACKAGE_URL "${CPACK_PACKAGE_HOMEPAGE_URL}")
  endif()

  # CPACK_RPM_PACKAGE_NAME (mandatory)
  if(NOT CPACK_RPM_PACKAGE_NAME)
    string(TOLOWER "${CPACK_PACKAGE_NAME}" CPACK_RPM_PACKAGE_NAME)
  endif()

  if(CPACK_RPM_PACKAGE_COMPONENT)
    string(TOUPPER "${CPACK_RPM_MAIN_COMPONENT}"
      CPACK_RPM_MAIN_COMPONENT_UPPER)

    if(NOT CPACK_RPM_MAIN_COMPONENT_UPPER STREQUAL CPACK_RPM_PACKAGE_COMPONENT_UPPER)
      string(APPEND CPACK_RPM_PACKAGE_NAME "-${CPACK_RPM_PACKAGE_COMPONENT_PART_NAME}")

      cpack_rpm_variable_fallback("CPACK_RPM_PACKAGE_NAME"
        "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_PACKAGE_NAME"
        "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_PACKAGE_NAME")
    endif()
  endif()

  # CPACK_RPM_PACKAGE_VERSION (mandatory)
  if(NOT CPACK_RPM_PACKAGE_VERSION)
    if(NOT CPACK_PACKAGE_VERSION)
      message(FATAL_ERROR "RPM package requires a package version")
    endif()
    set(CPACK_RPM_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
  endif()
  # Replace '-' in version with '_'
  # '-' character is  an Illegal RPM version character
  # it is illegal because it is used to separate
  # RPM "Version" from RPM "Release"
  string(REPLACE "-" "_" CPACK_RPM_PACKAGE_VERSION ${CPACK_RPM_PACKAGE_VERSION})

  # CPACK_RPM_PACKAGE_ARCHITECTURE (mandatory)
  if(NOT CPACK_RPM_PACKAGE_ARCHITECTURE)
    execute_process(COMMAND uname "-m"
                    OUTPUT_VARIABLE CPACK_RPM_PACKAGE_ARCHITECTURE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: using user-specified build arch = ${CPACK_RPM_PACKAGE_ARCHITECTURE}")
    endif()
  endif()

  if(CPACK_RPM_PACKAGE_COMPONENT)
    cpack_rpm_variable_fallback("CPACK_RPM_PACKAGE_ARCHITECTURE"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_PACKAGE_ARCHITECTURE"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_PACKAGE_ARCHITECTURE")

    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: using component build arch = ${CPACK_RPM_PACKAGE_ARCHITECTURE}")
    endif()
  endif()

  if(${CPACK_RPM_PACKAGE_ARCHITECTURE} STREQUAL "noarch")
    set(TMP_RPM_BUILDARCH "Buildarch: ${CPACK_RPM_PACKAGE_ARCHITECTURE}")
  else()
    set(TMP_RPM_BUILDARCH "")
  endif()

  # CPACK_RPM_PACKAGE_RELEASE
  # The RPM release is the numbering of the RPM package ITSELF
  # this is the version of the PACKAGING and NOT the version
  # of the CONTENT of the package.
  # You may well need to generate a new RPM package release
  # without changing the version of the packaged software.
  # This is the case when the packaging is buggy (not) the software :=)
  # If not set, 1 is a good candidate
  if(NOT CPACK_RPM_PACKAGE_RELEASE)
    set(CPACK_RPM_PACKAGE_RELEASE "1")
  endif()

  if(CPACK_RPM_PACKAGE_RELEASE_DIST)
    string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
  endif()

  # CPACK_RPM_PACKAGE_LICENSE
  if(NOT CPACK_RPM_PACKAGE_LICENSE)
    set(CPACK_RPM_PACKAGE_LICENSE "unknown")
  endif()

  # CPACK_RPM_PACKAGE_GROUP
  if(CPACK_RPM_PACKAGE_COMPONENT)
    cpack_rpm_variable_fallback("CPACK_RPM_PACKAGE_GROUP"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_PACKAGE_GROUP"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_PACKAGE_GROUP")
  endif()

  if(NOT CPACK_RPM_PACKAGE_GROUP)
    set(CPACK_RPM_PACKAGE_GROUP "unknown")
  endif()

  # CPACK_RPM_PACKAGE_VENDOR
  if(NOT CPACK_RPM_PACKAGE_VENDOR)
    if(CPACK_PACKAGE_VENDOR)
      set(CPACK_RPM_PACKAGE_VENDOR "${CPACK_PACKAGE_VENDOR}")
    else()
      set(CPACK_RPM_PACKAGE_VENDOR "unknown")
    endif()
  endif()

  # CPACK_RPM_PACKAGE_SOURCE
  # The name of the source tarball in case we generate a source RPM

  # CPACK_RPM_PACKAGE_DESCRIPTION
  # The variable content may be either
  #   - explicitly given by the user or
  #   - filled with the content of CPACK_PACKAGE_DESCRIPTION_FILE
  #     if it is defined
  #   - set to a default value
  #

  if(CPACK_RPM_PACKAGE_COMPONENT)
    cpack_rpm_variable_fallback("CPACK_RPM_PACKAGE_DESCRIPTION"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_PACKAGE_DESCRIPTION"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_PACKAGE_DESCRIPTION"
      "CPACK_COMPONENT_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_DESCRIPTION")
  endif()

  if(NOT CPACK_RPM_PACKAGE_DESCRIPTION)
    if(CPACK_PACKAGE_DESCRIPTION_FILE)
      file(READ ${CPACK_PACKAGE_DESCRIPTION_FILE} CPACK_RPM_PACKAGE_DESCRIPTION)
    else ()
      set(CPACK_RPM_PACKAGE_DESCRIPTION "no package description available")
    endif ()
  endif ()

  # CPACK_RPM_COMPRESSION_TYPE
  #
  if (CPACK_RPM_COMPRESSION_TYPE)
    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: User Specified RPM compression type: ${CPACK_RPM_COMPRESSION_TYPE}")
    endif()
    if(CPACK_RPM_COMPRESSION_TYPE STREQUAL "lzma")
      set(CPACK_RPM_COMPRESSION_TYPE_TMP "%define _binary_payload w9.lzdio")
    elseif(CPACK_RPM_COMPRESSION_TYPE STREQUAL "xz")
      if(CPACK_THREADS GREATER "0")
        set(CPACK_RPM_COMPRESSION_TYPE_TMP "%define _binary_payload w7T${CPACK_THREADS}.xzdio")
      else()
        set(CPACK_RPM_COMPRESSION_TYPE_TMP "%define _binary_payload w7T.xzdio")
      endif()
    elseif(CPACK_RPM_COMPRESSION_TYPE STREQUAL "bzip2")
      set(CPACK_RPM_COMPRESSION_TYPE_TMP "%define _binary_payload w9.bzdio")
    elseif(CPACK_RPM_COMPRESSION_TYPE STREQUAL "gzip")
      set(CPACK_RPM_COMPRESSION_TYPE_TMP "%define _binary_payload w9.gzdio")
    elseif(CPACK_RPM_COMPRESSION_TYPE STREQUAL "zstd")
      if(CPACK_THREADS GREATER "0")
        set(CPACK_RPM_COMPRESSION_TYPE_TMP "%define _binary_payload w19T${CPACK_THREADS}.zstdio")
      else()
        set(CPACK_RPM_COMPRESSION_TYPE_TMP "%define _binary_payload w19T0.zstdio")
      endif()
    else()
      message(FATAL_ERROR "Specified CPACK_RPM_COMPRESSION_TYPE value is not supported: ${CPACK_RPM_COMPRESSION_TYPE}")
    endif()
  else()
    set(CPACK_RPM_COMPRESSION_TYPE_TMP "")
  endif()

  if(NOT CPACK_RPM_PACKAGE_SOURCES)
    if(CPACK_PACKAGE_RELOCATABLE OR CPACK_RPM_PACKAGE_RELOCATABLE)
      if(CPACK_RPM_PACKAGE_DEBUG)
        message("CPackRPM:Debug: Trying to build a relocatable package")
      endif()
      if(CPACK_SET_DESTDIR AND (NOT CPACK_SET_DESTDIR STREQUAL "I_ON"))
        message("CPackRPM:Warning: CPACK_SET_DESTDIR is set (=${CPACK_SET_DESTDIR}) while requesting a relocatable package (CPACK_RPM_PACKAGE_RELOCATABLE is set): this is not supported, the package won't be relocatable.")
        set(CPACK_RPM_PACKAGE_RELOCATABLE FALSE)
      else()
        set(CPACK_RPM_PACKAGE_PREFIX ${CPACK_PACKAGING_INSTALL_PREFIX}) # kept for back compatibility (provided external RPM spec files)
        cpack_rpm_prepare_relocation_paths()
        set(CPACK_RPM_PACKAGE_RELOCATABLE TRUE)
      endif()
    endif()
  else()
    if(CPACK_RPM_PACKAGE_COMPONENT)
      message(FATAL_ERROR "CPACK_RPM_PACKAGE_SOURCES parameter can not be used"
        " in combination with CPACK_RPM_PACKAGE_COMPONENT parameter!")
    endif()

    set(CPACK_RPM_PACKAGE_RELOCATABLE FALSE) # disable relocatable option if building source RPM
  endif()

  execute_process(
    COMMAND "${RPMBUILD_EXECUTABLE}" --querytags
    OUTPUT_VARIABLE RPMBUILD_TAG_LIST
    RESULT_VARIABLE RPMBUILD_QUERYTAGS_SUCCESS
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  # In some versions of RPM, rpmbuild does not understand --querytags parameter,
  # but rpm does.
  if(NOT RPMBUILD_QUERYTAGS_SUCCESS EQUAL 0 AND RPM_EXECUTABLE)
    execute_process(
      COMMAND "${RPM_EXECUTABLE}" --querytags
      OUTPUT_VARIABLE RPMBUILD_TAG_LIST
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  string(REPLACE "\n" ";" RPMBUILD_TAG_LIST "${RPMBUILD_TAG_LIST}")

  # In some versions of RPM, weak dependency tags are present in the --querytags
  # list, but unsupported by rpmbuild. A different method must be used to check
  # if they are supported.
  if(RPM_EXECUTABLE)
    execute_process(
      COMMAND "${RPM_EXECUTABLE}" --suggests
      ERROR_QUIET
      RESULT_VARIABLE RPM_SUGGESTS_RESULT)
    if(NOT RPM_SUGGESTS_RESULT EQUAL 0)
      foreach(_WEAK_DEP SUGGESTS RECOMMENDS SUPPLEMENTS ENHANCES)
        list(REMOVE_ITEM RPMBUILD_TAG_LIST ${_WEAK_DEP})
      endforeach()
    endif()
  endif()

  if(CPACK_RPM_PACKAGE_EPOCH)
    set(TMP_RPM_EPOCH "Epoch: ${CPACK_RPM_PACKAGE_EPOCH}")
  endif()

  # Check if additional fields for RPM spec header are given
  # There may be some COMPONENT specific variables as well
  # If component specific var is not provided we use the global one
  # for each component
  foreach(_RPM_SPEC_HEADER URL REQUIRES SUGGESTS RECOMMENDS SUPPLEMENTS ENHANCES PROVIDES OBSOLETES PREFIX CONFLICTS AUTOPROV AUTOREQ AUTOREQPROV REQUIRES_PRE REQUIRES_POST REQUIRES_PREUN REQUIRES_POSTUN)

    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: processing ${_RPM_SPEC_HEADER}")
    endif()

    if(CPACK_RPM_PACKAGE_COMPONENT)
      cpack_rpm_variable_fallback("CPACK_RPM_PACKAGE_${_RPM_SPEC_HEADER}"
        "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_PACKAGE_${_RPM_SPEC_HEADER}"
        "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_PACKAGE_${_RPM_SPEC_HEADER}")
    endif()

    if(DEFINED CPACK_RPM_PACKAGE_${_RPM_SPEC_HEADER})
      # Prefix can be replaced by Prefixes but the old version still works so we'll ignore it for now
      # Requires* is a special case because it gets transformed to Requires(pre/post/preun/postun)
      # Auto* is a special case because the tags can not be queried by querytags rpmbuild flag
      set(special_case_tags_ PREFIX REQUIRES_PRE REQUIRES_POST REQUIRES_PREUN REQUIRES_POSTUN AUTOPROV AUTOREQ AUTOREQPROV)
      if(NOT _RPM_SPEC_HEADER IN_LIST RPMBUILD_TAG_LIST AND NOT _RPM_SPEC_HEADER IN_LIST special_case_tags_)
        message(AUTHOR_WARNING "CPackRPM:Warning: ${_RPM_SPEC_HEADER} not "
            "supported in provided rpmbuild. Tag will not be used.")
        continue()
      endif()

      if(CPACK_RPM_PACKAGE_DEBUG)
        message("CPackRPM:Debug: using CPACK_RPM_PACKAGE_${_RPM_SPEC_HEADER}")
      endif()

      set(CPACK_RPM_PACKAGE_${_RPM_SPEC_HEADER}_TMP ${CPACK_RPM_PACKAGE_${_RPM_SPEC_HEADER}})
    endif()

    # Treat the RPM Spec keyword iff it has been properly defined
    if(DEFINED CPACK_RPM_PACKAGE_${_RPM_SPEC_HEADER}_TMP)
      # Transform NAME --> Name e.g. PROVIDES --> Provides
      # The Upper-case first letter and lowercase tail is the
      # appropriate value required in the final RPM spec file.
      string(SUBSTRING ${_RPM_SPEC_HEADER} 1 -1 _PACKAGE_HEADER_TAIL)
      string(TOLOWER "${_PACKAGE_HEADER_TAIL}" _PACKAGE_HEADER_TAIL)
      string(SUBSTRING ${_RPM_SPEC_HEADER} 0 1 _PACKAGE_HEADER_NAME)
      string(APPEND _PACKAGE_HEADER_NAME "${_PACKAGE_HEADER_TAIL}")
      # The following keywords require parentheses around the "pre" or "post" suffix in the final RPM spec file.
      set(SCRIPTS_REQUIREMENTS_LIST REQUIRES_PRE REQUIRES_POST REQUIRES_PREUN REQUIRES_POSTUN)
      list(FIND SCRIPTS_REQUIREMENTS_LIST ${_RPM_SPEC_HEADER} IS_SCRIPTS_REQUIREMENT_FOUND)
      if(NOT ${IS_SCRIPTS_REQUIREMENT_FOUND} EQUAL -1)
        string(REPLACE "_" "(" _PACKAGE_HEADER_NAME "${_PACKAGE_HEADER_NAME}")
        string(APPEND _PACKAGE_HEADER_NAME ")")
      endif()
      if(CPACK_RPM_PACKAGE_DEBUG)
        message("CPackRPM:Debug: User defined ${_PACKAGE_HEADER_NAME}:\n ${CPACK_RPM_PACKAGE_${_RPM_SPEC_HEADER}_TMP}")
      endif()
      set(TMP_RPM_${_RPM_SPEC_HEADER} "${_PACKAGE_HEADER_NAME}: ${CPACK_RPM_PACKAGE_${_RPM_SPEC_HEADER}_TMP}")
      unset(CPACK_RPM_PACKAGE_${_RPM_SPEC_HEADER}_TMP)
    endif()
  endforeach()

  # CPACK_RPM_SPEC_INSTALL_POST
  # May be used to define a RPM post installation script
  # for example setting it to "/bin/true" may prevent
  # rpmbuild from stripping binaries.
  if(CPACK_RPM_SPEC_INSTALL_POST)
    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: User defined CPACK_RPM_SPEC_INSTALL_POST = ${CPACK_RPM_SPEC_INSTALL_POST}")
    endif()
    set(TMP_RPM_SPEC_INSTALL_POST "%define __spec_install_post ${CPACK_RPM_SPEC_INSTALL_POST}")
  endif()

  # CPACK_RPM_REQUIRES_EXCLUDE_FROM
  # May be defined to keep the dependency generator from
  # scanning specific files or directories for deps.
  if(CPACK_RPM_REQUIRES_EXCLUDE_FROM)
    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: User defined CPACK_RPM_REQUIRES_EXCLUDE_FROM = ${CPACK_RPM_REQUIRES_EXCLUDE_FROM}")
    endif()
    set(TMP_RPM_REQUIRES_EXCLUDE_FROM "%global __requires_exclude_from ${CPACK_RPM_REQUIRES_EXCLUDE_FROM}")
  endif()

  # CPACK_RPM_POST_INSTALL_SCRIPT_FILE (or CPACK_RPM_<COMPONENT>_POST_INSTALL_SCRIPT_FILE)
  # CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE (or CPACK_RPM_<COMPONENT>_POST_UNINSTALL_SCRIPT_FILE)
  # CPACK_RPM_POST_TRANS_SCRIPT_FILE (or CPACK_RPM_<COMPONENT>_POST_TRANS_SCRIPT_FILE)
  # May be used to embed a post installation/uninstallation/transaction script in the spec file.
  # The referred script file(s) will be read and directly
  # put after the %post or %postun or %posttrans section
  # ----------------------------------------------------------------
  # CPACK_RPM_PRE_INSTALL_SCRIPT_FILE (or CPACK_RPM_<COMPONENT>_PRE_INSTALL_SCRIPT_FILE)
  # CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE (or CPACK_RPM_<COMPONENT>_PRE_UNINSTALL_SCRIPT_FILE)
  # CPACK_RPM_PRE_TRANS_SCRIPT_FILE (or CPACK_RPM_<COMPONENT>_PRE_TRANS_SCRIPT_FILE)
  # May be used to embed a pre installation/uninstallation/transaction script in the spec file.
  # The referred script file(s) will be read and directly
  # put after the %pre or %preun or %pretrans section
  foreach(RPM_SCRIPT_FILE_TYPE_ "INSTALL" "UNINSTALL" "TRANS")
    foreach(RPM_SCRIPT_FILE_TIME_ "PRE" "POST")
      set("CPACK_RPM_${RPM_SCRIPT_FILE_TIME_}_${RPM_SCRIPT_FILE_TYPE_}_READ_FILE"
        "${CPACK_RPM_${RPM_SCRIPT_FILE_TIME_}_${RPM_SCRIPT_FILE_TYPE_}_SCRIPT_FILE}")

      if(CPACK_RPM_PACKAGE_COMPONENT)
        cpack_rpm_variable_fallback("CPACK_RPM_${RPM_SCRIPT_FILE_TIME_}_${RPM_SCRIPT_FILE_TYPE_}_READ_FILE"
          "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_${RPM_SCRIPT_FILE_TIME_}_${RPM_SCRIPT_FILE_TYPE_}_SCRIPT_FILE"
          "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_${RPM_SCRIPT_FILE_TIME_}_${RPM_SCRIPT_FILE_TYPE_}_SCRIPT_FILE")
      endif()

      # Handle file if it has been specified
      if(CPACK_RPM_${RPM_SCRIPT_FILE_TIME_}_${RPM_SCRIPT_FILE_TYPE_}_READ_FILE)
        if(EXISTS ${CPACK_RPM_${RPM_SCRIPT_FILE_TIME_}_${RPM_SCRIPT_FILE_TYPE_}_READ_FILE})
          file(READ ${CPACK_RPM_${RPM_SCRIPT_FILE_TIME_}_${RPM_SCRIPT_FILE_TYPE_}_READ_FILE}
            "CPACK_RPM_SPEC_${RPM_SCRIPT_FILE_TIME_}${RPM_SCRIPT_FILE_TYPE_}")
        else()
          message("CPackRPM:Warning: CPACK_RPM_${RPM_SCRIPT_FILE_TIME_}_${RPM_SCRIPT_FILE_TYPE_}_SCRIPT_FILE <${CPACK_RPM_${RPM_SCRIPT_FILE_TIME_}_${RPM_SCRIPT_FILE_TYPE_}_READ_FILE}> does not exist - ignoring")
        endif()
      else()
        # reset SPEC var value if no file has been specified
        # (either globally or component-wise)
        set("CPACK_RPM_SPEC_${RPM_SCRIPT_FILE_TIME_}${RPM_SCRIPT_FILE_TYPE_}" "")
      endif()
    endforeach()
  endforeach()

  # CPACK_RPM_CHANGELOG_FILE
  # May be used to embed a changelog in the spec file.
  # The referred file will be read and directly put after the %changelog section
  if(CPACK_RPM_CHANGELOG_FILE)
    if(EXISTS ${CPACK_RPM_CHANGELOG_FILE})
      file(READ ${CPACK_RPM_CHANGELOG_FILE} CPACK_RPM_SPEC_CHANGELOG)
    else()
      message(SEND_ERROR "CPackRPM:Warning: CPACK_RPM_CHANGELOG_FILE <${CPACK_RPM_CHANGELOG_FILE}> does not exist - ignoring")
    endif()
  else()
    set(CPACK_RPM_SPEC_CHANGELOG "* Sun Jul 4 2010 Eric Noulard <eric.noulard@gmail.com> - ${CPACK_RPM_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}\n  Generated by CPack RPM (no Changelog file were provided)")
  endif()

  # CPACK_RPM_SPEC_MORE_DEFINE
  # This is a generated spec rpm file spaceholder
  if(CPACK_RPM_SPEC_MORE_DEFINE)
    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: User defined more define spec line specified:\n ${CPACK_RPM_SPEC_MORE_DEFINE}")
    endif()
  endif()

  # Now we may create the RPM build tree structure
  set(CPACK_RPM_ROOTDIR "${CPACK_TOPLEVEL_DIRECTORY}")
  if(CPACK_RPM_PACKAGE_DEBUG)
    message("CPackRPM:Debug: Using CPACK_RPM_ROOTDIR=${CPACK_RPM_ROOTDIR}")
  endif()
  # Prepare RPM build tree
  file(MAKE_DIRECTORY ${CPACK_RPM_ROOTDIR})
  file(MAKE_DIRECTORY ${CPACK_RPM_ROOTDIR}/tmp)
  file(MAKE_DIRECTORY ${CPACK_RPM_ROOTDIR}/BUILD)
  file(MAKE_DIRECTORY ${CPACK_RPM_ROOTDIR}/RPMS)
  file(MAKE_DIRECTORY ${CPACK_RPM_ROOTDIR}/SOURCES)
  file(MAKE_DIRECTORY ${CPACK_RPM_ROOTDIR}/SPECS)
  file(MAKE_DIRECTORY ${CPACK_RPM_ROOTDIR}/SRPMS)

  # it seems rpmbuild can't handle spaces in the path
  # neither escaping (as below) nor putting quotes around the path seem to help
  #string(REGEX REPLACE " " "\\\\ " CPACK_RPM_DIRECTORY "${CPACK_TOPLEVEL_DIRECTORY}")
  set(CPACK_RPM_DIRECTORY "${CPACK_TOPLEVEL_DIRECTORY}")

  cpack_rpm_prepare_content_list()

  # In component case, put CPACK_ABSOLUTE_DESTINATION_FILES_<COMPONENT>
  #                   into CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL
  #         otherwise, put CPACK_ABSOLUTE_DESTINATION_FILES
  # This must be done BEFORE the CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL handling
  if(CPACK_RPM_PACKAGE_COMPONENT)
    if(CPACK_ABSOLUTE_DESTINATION_FILES)
      cpack_rpm_variable_fallback("COMPONENT_FILES_TAG"
        "CPACK_ABSOLUTE_DESTINATION_FILES_${CPACK_RPM_PACKAGE_COMPONENT}"
        "CPACK_ABSOLUTE_DESTINATION_FILES_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}")
      set(CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL "${${COMPONENT_FILES_TAG}}")
      if(CPACK_RPM_PACKAGE_DEBUG)
        message("CPackRPM:Debug: Handling Absolute Destination Files: <${CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL}>")
        message("CPackRPM:Debug: in component = ${CPACK_RPM_PACKAGE_COMPONENT}")
      endif()
    endif()
  else()
    if(CPACK_ABSOLUTE_DESTINATION_FILES)
      set(CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL "${CPACK_ABSOLUTE_DESTINATION_FILES}")
    endif()
  endif()

  # In component case, set CPACK_RPM_USER_FILELIST_INTERNAL with CPACK_RPM_<COMPONENT>_USER_FILELIST.
  set(CPACK_RPM_USER_FILELIST_INTERNAL "")
  if(CPACK_RPM_PACKAGE_COMPONENT)
    cpack_rpm_variable_fallback("CPACK_RPM_USER_FILELIST_INTERNAL"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_USER_FILELIST"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_USER_FILELIST")

    if(CPACK_RPM_PACKAGE_DEBUG AND CPACK_RPM_USER_FILELIST_INTERNAL)
      message("CPackRPM:Debug: Handling User Filelist: <${CPACK_RPM_USER_FILELIST_INTERNAL}>")
      message("CPackRPM:Debug: in component = ${CPACK_RPM_PACKAGE_COMPONENT}")
    endif()
  elseif(CPACK_RPM_USER_FILELIST)
    set(CPACK_RPM_USER_FILELIST_INTERNAL "${CPACK_RPM_USER_FILELIST}")
  endif()

  # Handle user specified file line list in CPACK_RPM_USER_FILELIST_INTERNAL
  # Remove those files from CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL
  #                      or CPACK_RPM_INSTALL_FILES,
  # hence it must be done before these auto-generated lists are processed.
  if(CPACK_RPM_USER_FILELIST_INTERNAL)
    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: Handling User Filelist: <${CPACK_RPM_USER_FILELIST_INTERNAL}>")
    endif()

    # Create CMake list from CPACK_RPM_INSTALL_FILES
    string(STRIP "${CPACK_RPM_INSTALL_FILES}" CPACK_RPM_INSTALL_FILES_LIST)
    string(REPLACE "\n" ";" CPACK_RPM_INSTALL_FILES_LIST
                            "${CPACK_RPM_INSTALL_FILES_LIST}")
    string(REPLACE "\"" "" CPACK_RPM_INSTALL_FILES_LIST
                            "${CPACK_RPM_INSTALL_FILES_LIST}")

    set(CPACK_RPM_USER_INSTALL_FILES "")
    foreach(F IN LISTS CPACK_RPM_USER_FILELIST_INTERNAL)
      string(REGEX REPLACE "%[A-Za-z]+(\\([^()]*\\))? " "" F_PATH ${F})
      string(REGEX MATCH "(%[A-Za-z]+(\\([^()]*\\))? )*" F_PREFIX ${F})
      string(STRIP ${F_PREFIX} F_PREFIX)

      if(CPACK_RPM_PACKAGE_DEBUG)
        message("CPackRPM:Debug: F_PREFIX=<${F_PREFIX}>, F_PATH=<${F_PATH}>")
      endif()
      if(F_PREFIX)
        string(APPEND F_PREFIX " ")
      endif()
      # Rebuild the user list file
      make_rpm_spec_path(F_SPEC "${F_PATH}")
      string(APPEND CPACK_RPM_USER_INSTALL_FILES "${F_PREFIX}${F_SPEC}\n")

      # Remove from CPACK_RPM_INSTALL_FILES and CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL
      list(REMOVE_ITEM CPACK_RPM_INSTALL_FILES_LIST ${F_PATH})
      # ABSOLUTE destination files list may not exists at all
      if (CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL)
        list(REMOVE_ITEM CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL ${F_PATH})
      endif()
    endforeach()

    # Rebuild CPACK_RPM_INSTALL_FILES
    set(CPACK_RPM_INSTALL_FILES "")
    foreach(F IN LISTS CPACK_RPM_INSTALL_FILES_LIST)
      make_rpm_spec_path(F_SPEC "${F}")
      string(APPEND CPACK_RPM_INSTALL_FILES "${F_SPEC}\n")
    endforeach()
  else()
    set(CPACK_RPM_USER_INSTALL_FILES "")
  endif()

  if (CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL)
    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: Handling Absolute Destination Files: ${CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL}")
    endif()
    # Remove trailing space
    string(STRIP "${CPACK_RPM_INSTALL_FILES}" CPACK_RPM_INSTALL_FILES_LIST)
    # Transform endline separated - string into CMake List
    string(REPLACE "\n" ";" CPACK_RPM_INSTALL_FILES_LIST "${CPACK_RPM_INSTALL_FILES_LIST}")
    # Remove unnecessary quotes
    string(REPLACE "\"" "" CPACK_RPM_INSTALL_FILES_LIST "${CPACK_RPM_INSTALL_FILES_LIST}")
    # Remove ABSOLUTE install file from INSTALL FILE LIST
    list(REMOVE_ITEM CPACK_RPM_INSTALL_FILES_LIST ${CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL})
    # Rebuild INSTALL_FILES
    set(CPACK_RPM_INSTALL_FILES "")
    foreach(F IN LISTS CPACK_RPM_INSTALL_FILES_LIST)
      make_rpm_spec_path(F_SPEC "${F}")
      string(APPEND CPACK_RPM_INSTALL_FILES "${F_SPEC}\n")
    endforeach()
    # Build ABSOLUTE_INSTALL_FILES
    set(CPACK_RPM_ABSOLUTE_INSTALL_FILES "")
    foreach(F IN LISTS CPACK_ABSOLUTE_DESTINATION_FILES_INTERNAL)
      make_rpm_spec_path(F_SPEC "${F}")
      string(APPEND CPACK_RPM_ABSOLUTE_INSTALL_FILES "%config ${F_SPEC}\n")
    endforeach()
    if(CPACK_RPM_PACKAGE_DEBUG)
      message("CPackRPM:Debug: CPACK_RPM_ABSOLUTE_INSTALL_FILES=${CPACK_RPM_ABSOLUTE_INSTALL_FILES}")
      message("CPackRPM:Debug: CPACK_RPM_INSTALL_FILES=${CPACK_RPM_INSTALL_FILES}")
    endif()
  else()
    # reset vars in order to avoid leakage of value(s) from one component to another
    set(CPACK_RPM_ABSOLUTE_INSTALL_FILES "")
  endif()

  cpack_rpm_variable_fallback("CPACK_RPM_DEBUGINFO_PACKAGE"
    "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_DEBUGINFO_PACKAGE"
    "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_DEBUGINFO_PACKAGE"
    "CPACK_RPM_DEBUGINFO_PACKAGE")
  if(CPACK_RPM_DEBUGINFO_PACKAGE OR (CPACK_RPM_DEBUGINFO_SINGLE_PACKAGE AND NOT GENERATE_SPEC_PARTS))
    cpack_rpm_variable_fallback("CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_BUILD_SOURCE_DIRS_PREFIX"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_BUILD_SOURCE_DIRS_PREFIX"
      "CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX")
    if(NOT CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX)
      set(CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX "/usr/src/debug/${CPACK_PACKAGE_FILE_NAME}${CPACK_RPM_PACKAGE_COMPONENT_PART_PATH}")
    endif()

    # handle cases where path contains extra slashes (e.g. /a//b/ instead of
    # /a/b)
    get_filename_component(CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX
      "${CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX}" ABSOLUTE)

    if(CPACK_RPM_DEBUGINFO_SINGLE_PACKAGE AND GENERATE_SPEC_PARTS)
      file(WRITE "${CPACK_RPM_ROOTDIR}/SPECS/${CPACK_RPM_PACKAGE_COMPONENT}.files"
        "${CPACK_RPM_INSTALL_FILES}")
    else()
      if(CPACK_RPM_DEBUGINFO_SINGLE_PACKAGE AND CPACK_RPM_PACKAGE_COMPONENT)
        # this part is only required by components packaging - with monolithic
        # packages we can be certain that there are no other components present
        # so CPACK_RPM_DEBUGINFO_SINGLE_PACKAGE is a noop
        if(CPACK_RPM_DEBUGINFO_PACKAGE)
          # only add current package files to debuginfo list if debuginfo
          # generation is enabled for current package
          string(STRIP "${CPACK_RPM_INSTALL_FILES}" install_files_)
          string(REPLACE "\n" ";" install_files_ "${install_files_}")
          string(REPLACE "\"" "" install_files_ "${install_files_}")
        else()
          unset(install_files_)
        endif()

        file(GLOB files_ "${CPACK_RPM_DIRECTORY}/SPECS/*.files")

        foreach(f_ IN LISTS files_)
          file(READ "${f_}" tmp_)
          string(APPEND install_files_ ";${tmp_}")
        endforeach()

        # if there were other components/groups so we need to move files from them
        # to current component otherwise those files won't be found
        file(GLOB components_ LIST_DIRECTORIES true RELATIVE
          "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}"
          "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}/*")
        foreach(component_ IN LISTS components_)
          string(TOUPPER "${component_}" component_dir_upper_)
          if(component_dir_upper_ STREQUAL CPACK_RPM_PACKAGE_COMPONENT_UPPER)
            # skip current component
            continue()
          endif()

          file(GLOB_RECURSE files_for_move_ LIST_DIRECTORIES true RELATIVE
            "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}/${component_}"
            "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}/${component_}/*")

          foreach(f_ IN LISTS files_for_move_)
            set(src_file_
              "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}/${component_}/${f_}")

            if(IS_DIRECTORY "${src_file_}")
              file(MAKE_DIRECTORY "${WDIR}/${f_}")
              continue()
            endif()

            get_filename_component(dir_path_ "${f_}" DIRECTORY)

            # check that we are not overriding an existing file that doesn't
            # match the file that we want to copy
            if(EXISTS "${src_file_}" AND EXISTS "${WDIR}/${f_}")
              execute_process(
                  COMMAND ${CMAKE_COMMAND} -E compare_files "${src_file_}" "${WDIR}/${f_}"
                  RESULT_VARIABLE res_
                )
              if(res_)
                message(FATAL_ERROR "CPackRPM:Error: File on path '${WDIR}/${f_}'"
                  " already exists but is a different than the one in component"
                  " '${component_}'! Packages will not be generated.")
              endif()
            endif()

            file(MAKE_DIRECTORY "${WDIR}/${dir_path_}")
            file(RENAME "${src_file_}"
              "${WDIR}/${f_}")
          endforeach()
        endforeach()

        cpack_rpm_debugsymbol_check("${install_files_}" "${WDIR}")
      else()
        string(STRIP "${CPACK_RPM_INSTALL_FILES}" install_files_)
        string(REPLACE "\n" ";" install_files_ "${install_files_}")
        string(REPLACE "\"" "" install_files_ "${install_files_}")

        cpack_rpm_debugsymbol_check("${install_files_}" "${WDIR}")
      endif()

      if(TMP_DEBUGINFO_ADDITIONAL_SOURCES)
        set(TMP_RPM_DEBUGINFO "
# Modified version of %%debug_package macro
# defined in /usr/lib/rpm/macros as that one
# can't handle injection of extra source files.
%ifnarch noarch
%global __debug_package 1
%package debuginfo
Summary: Debug information for package %{name}
Group: Development/Debug
AutoReqProv: 0
%description debuginfo
This package provides debug information for package %{name}.
Debug information is useful when developing applications that use this
package or when debugging this package.
%files debuginfo -f debugfiles.list
%defattr(-,root,root)
${TMP_DEBUGINFO_ADDITIONAL_SOURCES}
%endif
")
      elseif(CPACK_RPM_DEBUGINFO_SINGLE_PACKAGE)
        message(AUTHOR_WARNING "CPackRPM:Warning: debuginfo package was requested"
          " but will not be generated as no source files were found!")
      else()
        message(AUTHOR_WARNING "CPackRPM:Warning: debuginfo package was requested"
          " but will not be generated as no source files were found! Component: '"
          "${CPACK_RPM_PACKAGE_COMPONENT}'.")
      endif()
    endif()
  endif()

  # Prepare install files
  cpack_rpm_prepare_install_files(
      "${CPACK_RPM_INSTALL_FILES}"
      "${WDIR}"
      "${RPM_USED_PACKAGE_PREFIXES}"
      "${CPACK_RPM_PACKAGE_RELOCATABLE}"
    )

  # set default user and group
  foreach(_PERM_TYPE "USER" "GROUP")
    if(CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_DEFAULT_${_PERM_TYPE})
      set(TMP_DEFAULT_${_PERM_TYPE} "${CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_DEFAULT_${_PERM_TYPE}}")
    elseif(CPACK_RPM_DEFAULT_${_PERM_TYPE})
      set(TMP_DEFAULT_${_PERM_TYPE} "${CPACK_RPM_DEFAULT_${_PERM_TYPE}}")
    else()
      set(TMP_DEFAULT_${_PERM_TYPE} "root")
    endif()
  endforeach()

  # set default file and dir permissions
  foreach(_PERM_TYPE "FILE" "DIR")
    if(CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_DEFAULT_${_PERM_TYPE}_PERMISSIONS)
      get_unix_permissions_octal_notation("CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_DEFAULT_${_PERM_TYPE}_PERMISSIONS" "TMP_DEFAULT_${_PERM_TYPE}_PERMISSIONS")
      set(_PERMISSIONS_VAR "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_DEFAULT_${_PERM_TYPE}_PERMISSIONS")
    elseif(CPACK_RPM_DEFAULT_${_PERM_TYPE}_PERMISSIONS)
      get_unix_permissions_octal_notation("CPACK_RPM_DEFAULT_${_PERM_TYPE}_PERMISSIONS" "TMP_DEFAULT_${_PERM_TYPE}_PERMISSIONS")
      set(_PERMISSIONS_VAR "CPACK_RPM_DEFAULT_${_PERM_TYPE}_PERMISSIONS")
    else()
      set(TMP_DEFAULT_${_PERM_TYPE}_PERMISSIONS "-")
    endif()
  endforeach()

  # The name of the final spec file to be used by rpmbuild
  set(CPACK_RPM_BINARY_SPECFILE "${CPACK_RPM_ROOTDIR}/SPECS/${CPACK_RPM_PACKAGE_NAME}.spec")

  # Print out some debug information if we were asked for that
  if(CPACK_RPM_PACKAGE_DEBUG)
     message("CPackRPM:Debug: CPACK_TOPLEVEL_DIRECTORY          = ${CPACK_TOPLEVEL_DIRECTORY}")
     message("CPackRPM:Debug: CPACK_TOPLEVEL_TAG                = ${CPACK_TOPLEVEL_TAG}")
     message("CPackRPM:Debug: CPACK_TEMPORARY_DIRECTORY         = ${CPACK_TEMPORARY_DIRECTORY}")
     message("CPackRPM:Debug: CPACK_OUTPUT_FILE_NAME            = ${CPACK_OUTPUT_FILE_NAME}")
     message("CPackRPM:Debug: CPACK_OUTPUT_FILE_PATH            = ${CPACK_OUTPUT_FILE_PATH}")
     message("CPackRPM:Debug: CPACK_PACKAGE_FILE_NAME           = ${CPACK_PACKAGE_FILE_NAME}")
     message("CPackRPM:Debug: CPACK_RPM_BINARY_SPECFILE         = ${CPACK_RPM_BINARY_SPECFILE}")
     message("CPackRPM:Debug: CPACK_PACKAGE_INSTALL_DIRECTORY   = ${CPACK_PACKAGE_INSTALL_DIRECTORY}")
     message("CPackRPM:Debug: CPACK_TEMPORARY_PACKAGE_FILE_NAME = ${CPACK_TEMPORARY_PACKAGE_FILE_NAME}")
  endif()

  #
  # USER generated/provided spec file handling.
  #

  # We can have a component specific spec file.
  if(CPACK_RPM_PACKAGE_COMPONENT)
    cpack_rpm_variable_fallback("CPACK_RPM_USER_BINARY_SPECFILE"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_USER_BINARY_SPECFILE"
      "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_USER_BINARY_SPECFILE")
  endif()

  cpack_rpm_variable_fallback("CPACK_RPM_FILE_NAME"
    "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_FILE_NAME"
    "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_FILE_NAME"
    "CPACK_RPM_FILE_NAME")
  if(NOT CPACK_RPM_FILE_NAME STREQUAL "RPM-DEFAULT")
    if(CPACK_RPM_FILE_NAME)
      if(NOT CPACK_RPM_FILE_NAME MATCHES ".*\\.rpm")
        set(CPACK_RPM_FILE_NAME "${CPACK_RPM_FILE_NAME}.rpm")
      endif()
    else()
      # old file name format for back compatibility
      string(TOUPPER "${CPACK_RPM_MAIN_COMPONENT}"
        CPACK_RPM_MAIN_COMPONENT_UPPER)

      if(CPACK_RPM_MAIN_COMPONENT_UPPER STREQUAL CPACK_RPM_PACKAGE_COMPONENT_UPPER)
        # this is the main component so ignore the component filename part
        set(CPACK_RPM_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.rpm")
      else()
        set(CPACK_RPM_FILE_NAME "${CPACK_OUTPUT_FILE_NAME}")
      endif()
    endif()
    # else example:
    #set(CPACK_RPM_FILE_NAME "${CPACK_RPM_PACKAGE_NAME}-${CPACK_RPM_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}-${CPACK_RPM_PACKAGE_ARCHITECTURE}.rpm")

    if(CPACK_RPM_DEBUGINFO_SINGLE_PACKAGE AND GENERATE_SPEC_PARTS)
      string(TOLOWER "${CPACK_RPM_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}.*\\.rpm" expected_filename_)

      file(WRITE "${CPACK_RPM_ROOTDIR}/SPECS/${CPACK_RPM_PACKAGE_COMPONENT}.rpm_name"
        "${expected_filename_};${CPACK_RPM_FILE_NAME}")
    elseif(NOT CPACK_RPM_DEBUGINFO_PACKAGE)
      set(FILE_NAME_DEFINE "%define _rpmfilename ${CPACK_RPM_FILE_NAME}")
    endif()
  endif()

  if(CPACK_RPM_PACKAGE_SOURCES) # source rpm
    set(archive_name_ "${CPACK_RPM_PACKAGE_NAME}-${CPACK_RPM_PACKAGE_VERSION}")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar "cfvz" "${CPACK_RPM_DIRECTORY}/SOURCES/${archive_name_}.tar.gz" "${CPACK_PACKAGE_FILE_NAME}"
        WORKING_DIRECTORY ${CPACK_RPM_DIRECTORY}
      )
    set(TMP_RPM_SOURCE "Source: ${archive_name_}.tar.gz")

    if(CPACK_RPM_BUILDREQUIRES)
      set(TMP_RPM_BUILD_REQUIRES "BuildRequires: ${CPACK_RPM_BUILDREQUIRES}")
    endif()

    # Disable debuginfo packages - srpm generates invalid packages due to
    # releasing control to cpack to generate binary packages.
    # Note however that this doesn't prevent cpack to generate debuginfo
    # packages when run from srpm with --rebuild.
    set(TMP_RPM_DISABLE_DEBUGINFO "%define debug_package %{nil}")

    if(NOT CPACK_RPM_SOURCE_PKG_PACKAGING_INSTALL_PREFIX)
      set(CPACK_RPM_SOURCE_PKG_PACKAGING_INSTALL_PREFIX "/")
    endif()

    set(TMP_RPM_BUILD
      "
%build
mkdir cpack_rpm_build_dir
cd cpack_rpm_build_dir
'${CMAKE_COMMAND}' ${CPACK_RPM_SOURCE_PKG_BUILD_PARAMS} -DCPACK_PACKAGING_INSTALL_PREFIX=${CPACK_RPM_SOURCE_PKG_PACKAGING_INSTALL_PREFIX} ../${CPACK_PACKAGE_FILE_NAME}
make %{?_smp_mflags}" # %{?_smp_mflags} -> -j option
      )
    set(TMP_RPM_INSTALL
      "
cd cpack_rpm_build_dir
cpack -G RPM
mv *.rpm %_rpmdir"
      )
    set(TMP_RPM_PREP "%setup -c")

    set(RPMBUILD_FLAGS "-bs")

     file(WRITE ${CPACK_RPM_BINARY_SPECFILE}.in
      "# Restore old style debuginfo creation for rpm >= 4.14.
%undefine _debugsource_packages
%undefine _debuginfo_subpackages

# -*- rpm-spec -*-
BuildRoot:      %_topdir/\@CPACK_PACKAGE_FILE_NAME\@
Summary:        \@CPACK_RPM_PACKAGE_SUMMARY\@
Name:           \@CPACK_RPM_PACKAGE_NAME\@
Version:        \@CPACK_RPM_PACKAGE_VERSION\@
Release:        \@CPACK_RPM_PACKAGE_RELEASE\@
License:        \@CPACK_RPM_PACKAGE_LICENSE\@
Group:          \@CPACK_RPM_PACKAGE_GROUP\@
Vendor:         \@CPACK_RPM_PACKAGE_VENDOR\@

\@TMP_RPM_SOURCE\@
\@TMP_RPM_BUILD_REQUIRES\@
\@TMP_RPM_BUILDARCH\@
\@TMP_RPM_PREFIXES\@

\@TMP_RPM_DISABLE_DEBUGINFO\@

%define _rpmdir %_topdir/RPMS
%define _srcrpmdir %_topdir/SRPMS
\@FILE_NAME_DEFINE\@
%define _unpackaged_files_terminate_build 0
\@TMP_RPM_SPEC_INSTALL_POST\@
\@TMP_RPM_REQUIRES_EXCLUDE_FROM\@
\@CPACK_RPM_SPEC_MORE_DEFINE\@
\@CPACK_RPM_COMPRESSION_TYPE_TMP\@

%description
\@CPACK_RPM_PACKAGE_DESCRIPTION\@

# This is a shortcutted spec file generated by CMake RPM generator
# we skip _install step because CPack does that for us.
# We do only save CPack installed tree in _prepr
# and then restore it in build.
%prep
\@TMP_RPM_PREP\@

\@TMP_RPM_BUILD\@

#p build

%install
\@TMP_RPM_INSTALL\@

%clean

%changelog
\@CPACK_RPM_SPEC_CHANGELOG\@
"
    )

  elseif(GENERATE_SPEC_PARTS) # binary rpm with single debuginfo package

    set_spec_scripts("${CPACK_RPM_PACKAGE_NAME}")

    file(WRITE ${CPACK_RPM_BINARY_SPECFILE}.in
        "# -*- rpm-spec -*-
%package -n \@CPACK_RPM_PACKAGE_NAME\@
Summary:        \@CPACK_RPM_PACKAGE_SUMMARY\@
Version:        \@CPACK_RPM_PACKAGE_VERSION\@
Release:        \@CPACK_RPM_PACKAGE_RELEASE\@
License:        \@CPACK_RPM_PACKAGE_LICENSE\@
Group:          \@CPACK_RPM_PACKAGE_GROUP\@
Vendor:         \@CPACK_RPM_PACKAGE_VENDOR\@

\@TMP_RPM_URL\@
\@TMP_RPM_REQUIRES\@
\@TMP_RPM_REQUIRES_PRE\@
\@TMP_RPM_REQUIRES_POST\@
\@TMP_RPM_REQUIRES_PREUN\@
\@TMP_RPM_REQUIRES_POSTUN\@
\@TMP_RPM_PROVIDES\@
\@TMP_RPM_OBSOLETES\@
\@TMP_RPM_CONFLICTS\@
\@TMP_RPM_RECOMMENDS\@
\@TMP_RPM_SUGGESTS\@
\@TMP_RPM_SUPPLEMENTS\@
\@TMP_RPM_ENHANCES\@
\@TMP_RPM_AUTOPROV\@
\@TMP_RPM_AUTOREQ\@
\@TMP_RPM_AUTOREQPROV\@
\@TMP_RPM_BUILDARCH\@
\@TMP_RPM_PREFIXES\@
\@TMP_RPM_EPOCH\@

%description -n \@CPACK_RPM_PACKAGE_NAME\@
\@CPACK_RPM_PACKAGE_DESCRIPTION\@

\@post_\@
\@posttrans_\@
\@postun_\@
\@pre_\@
\@pretrans_\@
\@preun_\@

%files -n \@CPACK_RPM_PACKAGE_NAME\@
%defattr(\@TMP_DEFAULT_FILE_PERMISSIONS\@,\@TMP_DEFAULT_USER\@,\@TMP_DEFAULT_GROUP\@,\@TMP_DEFAULT_DIR_PERMISSIONS\@)
\@CPACK_RPM_INSTALL_FILES\@
\@CPACK_RPM_ABSOLUTE_INSTALL_FILES\@
\@CPACK_RPM_USER_INSTALL_FILES\@
"
    )

  else()  # binary rpm
    if(CPACK_RPM_DEBUGINFO_SINGLE_PACKAGE)
      # find generated spec file and take its name
      file(GLOB spec_files_ "${CPACK_RPM_DIRECTORY}/SPECS/*.spec")

      foreach(f_ IN LISTS spec_files_)
        file(READ "${f_}" tmp_)
        string(APPEND TMP_OTHER_COMPONENTS "\n${tmp_}\n")
      endforeach()
    endif()

    # We should generate a USER spec file template:
    #  - either because the user asked for it : CPACK_RPM_GENERATE_USER_BINARY_SPECFILE_TEMPLATE
    #  - or the user did not provide one : NOT CPACK_RPM_USER_BINARY_SPECFILE
    set(RPMBUILD_FLAGS "-bb")
    if(CPACK_RPM_GENERATE_USER_BINARY_SPECFILE_TEMPLATE OR NOT CPACK_RPM_USER_BINARY_SPECFILE)

      set_spec_scripts("")

      file(WRITE ${CPACK_RPM_BINARY_SPECFILE}.in
        "# Restore old style debuginfo creation for rpm >= 4.14.
%undefine _debugsource_packages
%undefine _debuginfo_subpackages

# -*- rpm-spec -*-
BuildRoot:      %_topdir/\@CPACK_PACKAGE_FILE_NAME\@\@CPACK_RPM_PACKAGE_COMPONENT_PART_PATH\@
Summary:        \@CPACK_RPM_PACKAGE_SUMMARY\@
Name:           \@CPACK_RPM_PACKAGE_NAME\@
Version:        \@CPACK_RPM_PACKAGE_VERSION\@
Release:        \@CPACK_RPM_PACKAGE_RELEASE\@
License:        \@CPACK_RPM_PACKAGE_LICENSE\@
Group:          \@CPACK_RPM_PACKAGE_GROUP\@
Vendor:         \@CPACK_RPM_PACKAGE_VENDOR\@

\@TMP_RPM_URL\@
\@TMP_RPM_REQUIRES\@
\@TMP_RPM_REQUIRES_PRE\@
\@TMP_RPM_REQUIRES_POST\@
\@TMP_RPM_REQUIRES_PREUN\@
\@TMP_RPM_REQUIRES_POSTUN\@
\@TMP_RPM_PROVIDES\@
\@TMP_RPM_OBSOLETES\@
\@TMP_RPM_CONFLICTS\@
\@TMP_RPM_RECOMMENDS\@
\@TMP_RPM_SUGGESTS\@
\@TMP_RPM_SUPPLEMENTS\@
\@TMP_RPM_ENHANCES\@
\@TMP_RPM_AUTOPROV\@
\@TMP_RPM_AUTOREQ\@
\@TMP_RPM_AUTOREQPROV\@
\@TMP_RPM_BUILDARCH\@
\@TMP_RPM_PREFIXES\@
\@TMP_RPM_EPOCH\@

\@TMP_RPM_DEBUGINFO\@

%define _rpmdir %_topdir/RPMS
%define _srcrpmdir %_topdir/SRPMS
\@FILE_NAME_DEFINE\@
%define _unpackaged_files_terminate_build 0
\@TMP_RPM_SPEC_INSTALL_POST\@
\@TMP_RPM_REQUIRES_EXCLUDE_FROM\@
\@CPACK_RPM_SPEC_MORE_DEFINE\@
\@CPACK_RPM_COMPRESSION_TYPE_TMP\@

%description
\@CPACK_RPM_PACKAGE_DESCRIPTION\@

# This is a shortcutted spec file generated by CMake RPM generator
# we skip _install step because CPack does that for us.
# We do only save CPack installed tree in _prepr
# and then restore it in build.
%prep
mv $RPM_BUILD_ROOT %_topdir/tmpBBroot

%install
if [ -e $RPM_BUILD_ROOT ];
then
  rm -rf $RPM_BUILD_ROOT
fi
mv %_topdir/tmpBBroot $RPM_BUILD_ROOT

\@TMP_RPM_DEBUGINFO_INSTALL\@

%clean

\@post_\@
\@posttrans_\@
\@postun_\@
\@pre_\@
\@pretrans_\@
\@preun_\@

%files
%defattr(\@TMP_DEFAULT_FILE_PERMISSIONS\@,\@TMP_DEFAULT_USER\@,\@TMP_DEFAULT_GROUP\@,\@TMP_DEFAULT_DIR_PERMISSIONS\@)
\@CPACK_RPM_INSTALL_FILES\@
\@CPACK_RPM_ABSOLUTE_INSTALL_FILES\@
\@CPACK_RPM_USER_INSTALL_FILES\@

%changelog
\@CPACK_RPM_SPEC_CHANGELOG\@

\@TMP_OTHER_COMPONENTS\@
"
      )
    endif()

    # Stop here if we were asked to only generate a template USER spec file
    # The generated file may then be used as a template by user who wants
    # to customize their own spec file.
    if(CPACK_RPM_GENERATE_USER_BINARY_SPECFILE_TEMPLATE)
      message(FATAL_ERROR "CPackRPM: STOP here Generated USER binary spec file template is: ${CPACK_RPM_BINARY_SPECFILE}.in")
    endif()
  endif()

  # After that we may either use a user provided spec file
  # or generate one using appropriate variables value.
  if(CPACK_RPM_USER_BINARY_SPECFILE)
    # User may have specified SPECFILE just use it
    message("CPackRPM: Will use USER specified spec file: ${CPACK_RPM_USER_BINARY_SPECFILE}")
    # The user provided file is processed for @var replacement
    configure_file(${CPACK_RPM_USER_BINARY_SPECFILE} ${CPACK_RPM_BINARY_SPECFILE} @ONLY)
  else()
    # No User specified spec file, will use the generated spec file
    message("CPackRPM: Will use GENERATED spec file: ${CPACK_RPM_BINARY_SPECFILE}")
    # Note the just created file is processed for @var replacement
    configure_file(${CPACK_RPM_BINARY_SPECFILE}.in ${CPACK_RPM_BINARY_SPECFILE} @ONLY)
  endif()

  if(NOT GENERATE_SPEC_PARTS) # generate package
    # Now call rpmbuild using the SPECFILE
    execute_process(
      COMMAND "${RPMBUILD_EXECUTABLE}" ${RPMBUILD_FLAGS}
              --define "_topdir ${CPACK_RPM_DIRECTORY}"
              --buildroot "%_topdir/${CPACK_PACKAGE_FILE_NAME}${CPACK_RPM_PACKAGE_COMPONENT_PART_PATH}"
              --target "${CPACK_RPM_PACKAGE_ARCHITECTURE}"
              "${CPACK_RPM_BINARY_SPECFILE}"
      WORKING_DIRECTORY "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}${CPACK_RPM_PACKAGE_COMPONENT_PART_PATH}"
      RESULT_VARIABLE CPACK_RPMBUILD_EXEC_RESULT
      ERROR_FILE "${CPACK_TOPLEVEL_DIRECTORY}/rpmbuild${CPACK_RPM_PACKAGE_NAME}.err"
      OUTPUT_FILE "${CPACK_TOPLEVEL_DIRECTORY}/rpmbuild${CPACK_RPM_PACKAGE_NAME}.out")
    if(CPACK_RPM_PACKAGE_DEBUG OR CPACK_RPMBUILD_EXEC_RESULT)
      file(READ ${CPACK_TOPLEVEL_DIRECTORY}/rpmbuild${CPACK_RPM_PACKAGE_NAME}.err RPMBUILDERR)
      file(READ ${CPACK_TOPLEVEL_DIRECTORY}/rpmbuild${CPACK_RPM_PACKAGE_NAME}.out RPMBUILDOUT)
      message("CPackRPM:Debug: You may consult rpmbuild logs in: ")
      message("CPackRPM:Debug:    - ${CPACK_TOPLEVEL_DIRECTORY}/rpmbuild${CPACK_RPM_PACKAGE_NAME}.err")
      message("CPackRPM:Debug: *** ${RPMBUILDERR} ***")
      message("CPackRPM:Debug:    - ${CPACK_TOPLEVEL_DIRECTORY}/rpmbuild${CPACK_RPM_PACKAGE_NAME}.out")
      message("CPackRPM:Debug: *** ${RPMBUILDOUT} ***")
    endif()

    # find generated rpm files and take their names
    file(GLOB_RECURSE GENERATED_FILES "${CPACK_RPM_DIRECTORY}/RPMS/*.rpm"
      "${CPACK_RPM_DIRECTORY}/SRPMS/*.rpm")

    if(NOT GENERATED_FILES)
      message(FATAL_ERROR "RPM package was not generated! ${CPACK_RPM_DIRECTORY}")
    endif()

    unset(expected_filenames_)
    unset(filenames_)
    if(CPACK_RPM_DEBUGINFO_PACKAGE AND NOT CPACK_RPM_FILE_NAME STREQUAL "RPM-DEFAULT")
      list(APPEND expected_filenames_
        "${CPACK_RPM_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}.*\\.rpm")
      list(APPEND filenames_ "${CPACK_RPM_FILE_NAME}")
    endif()

    if(CPACK_RPM_DEBUGINFO_PACKAGE)
      cpack_rpm_variable_fallback("CPACK_RPM_DEBUGINFO_FILE_NAME"
        "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT}_DEBUGINFO_FILE_NAME"
        "CPACK_RPM_${CPACK_RPM_PACKAGE_COMPONENT_UPPER}_DEBUGINFO_FILE_NAME"
        "CPACK_RPM_DEBUGINFO_FILE_NAME")

      if(CPACK_RPM_DEBUGINFO_FILE_NAME AND
        NOT CPACK_RPM_DEBUGINFO_FILE_NAME STREQUAL "RPM-DEFAULT")
        list(APPEND expected_filenames_
          "${CPACK_RPM_PACKAGE_NAME}-debuginfo-${CPACK_PACKAGE_VERSION}.*\\.rpm")
        string(REPLACE "@cpack_component@" "${CPACK_RPM_PACKAGE_COMPONENT}"
          CPACK_RPM_DEBUGINFO_FILE_NAME "${CPACK_RPM_DEBUGINFO_FILE_NAME}")
        list(APPEND filenames_ "${CPACK_RPM_DEBUGINFO_FILE_NAME}")
      endif()
    endif()

    # check if other files have to be renamed
    file(GLOB rename_files_ "${CPACK_RPM_DIRECTORY}/SPECS/*.rpm_name")
    if(rename_files_)
      foreach(f_ IN LISTS rename_files_)
        file(READ "${f_}" tmp_)
        list(GET tmp_ 0 efn_)
        list(APPEND expected_filenames_ "${efn_}")
        list(GET tmp_ 1 fn_)
        list(APPEND filenames_ "${fn_}")
      endforeach()
    endif()

    if(expected_filenames_)
      foreach(F IN LISTS GENERATED_FILES)
        unset(matched_)
        foreach(expected_ IN LISTS expected_filenames_)
          if(F MATCHES ".*/${expected_}")
            list(FIND expected_filenames_ "${expected_}" idx_)
            list(GET filenames_ ${idx_} filename_)
            get_filename_component(FILE_PATH "${F}" DIRECTORY)
            file(RENAME "${F}" "${FILE_PATH}/${filename_}")
            list(APPEND new_files_list_ "${FILE_PATH}/${filename_}")
            set(matched_ "YES")

            break()
          endif()
        endforeach()

        if(NOT matched_)
          list(APPEND new_files_list_ "${F}")
        endif()
      endforeach()

      set(GENERATED_FILES "${new_files_list_}")
    endif()
  endif()

  set(GEN_CPACK_OUTPUT_FILES "${GENERATED_FILES}" PARENT_SCOPE)

  if(CPACK_RPM_PACKAGE_DEBUG)
     message("CPackRPM:Debug: GEN_CPACK_OUTPUT_FILES = ${GENERATED_FILES}")
  endif()
endfunction()

cpack_rpm_generate_package()
