# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This file contains KDE 3 macros. See FindKDE3.cmake for documentation.
# Author: neundorf@kde.org

# Included for backward compatibility, otherwise unused.
include(AddFileDependencies)

macro(KDE3_ADD_DCOP_SKELS _sources)
  foreach (_current_FILE ${ARGN})

    get_filename_component(_tmp_FILE ${_current_FILE} ABSOLUTE)
    get_filename_component(_basename ${_tmp_FILE} NAME_WE)

    set(_skel ${CMAKE_CURRENT_BINARY_DIR}/${_basename}_skel.cpp)
    set(_kidl ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.kidl)

    if (NOT HAVE_${_basename}_KIDL_RULE)
      set(HAVE_${_basename}_KIDL_RULE ON)

      add_custom_command(OUTPUT ${_kidl}
        COMMAND ${KDE3_DCOPIDL_EXECUTABLE}
        ARGS ${_tmp_FILE} > ${_kidl}
        DEPENDS ${_tmp_FILE}
      )

    endif ()

    if (NOT HAVE_${_basename}_SKEL_RULE)
      set(HAVE_${_basename}_SKEL_RULE ON)

      add_custom_command(OUTPUT ${_skel}
        COMMAND ${KDE3_DCOPIDL2CPP_EXECUTABLE}
        ARGS --c++-suffix cpp --no-signals --no-stub ${_kidl}
        DEPENDS ${_kidl}
      )

    endif ()

    set(${_sources} ${${_sources}} ${_skel})

  endforeach ()

endmacro()


macro(KDE3_ADD_DCOP_STUBS _sources)
  foreach (_current_FILE ${ARGN})

    get_filename_component(_tmp_FILE ${_current_FILE} ABSOLUTE)

    get_filename_component(_basename ${_tmp_FILE} NAME_WE)

    set(_stub_CPP ${CMAKE_CURRENT_BINARY_DIR}/${_basename}_stub.cpp)
    set(_kidl ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.kidl)

    if (NOT HAVE_${_basename}_KIDL_RULE)
      set(HAVE_${_basename}_KIDL_RULE ON)


      add_custom_command(OUTPUT ${_kidl}
        COMMAND ${KDE3_DCOPIDL_EXECUTABLE}
        ARGS ${_tmp_FILE} > ${_kidl}
        DEPENDS ${_tmp_FILE}
      )

    endif ()


    if (NOT HAVE_${_basename}_STUB_RULE)
      set(HAVE_${_basename}_STUB_RULE ON)

      add_custom_command(OUTPUT ${_stub_CPP}
        COMMAND ${KDE3_DCOPIDL2CPP_EXECUTABLE}
        ARGS --c++-suffix cpp --no-signals --no-skel ${_kidl}
        DEPENDS ${_kidl}
      )

    endif ()

    set(${_sources} ${${_sources}} ${_stub_CPP})

  endforeach ()

endmacro()


macro(KDE3_ADD_KCFG_FILES _sources)
  foreach (_current_FILE ${ARGN})

    get_filename_component(_tmp_FILE ${_current_FILE} ABSOLUTE)

    get_filename_component(_basename ${_tmp_FILE} NAME_WE)

    file(READ ${_tmp_FILE} _contents)
    string(REGEX REPLACE "^(.*\n)?File=([^\n]+)\n.*$" "\\2"  _kcfg_FILE "${_contents}")

    set(_src_FILE    ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.cpp)
    set(_header_FILE ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.h)

    add_custom_command(OUTPUT ${_src_FILE}
      COMMAND ${KDE3_KCFGC_EXECUTABLE}
      ARGS ${CMAKE_CURRENT_SOURCE_DIR}/${_kcfg_FILE} ${_tmp_FILE}
      DEPENDS ${_tmp_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${_kcfg_FILE}
    )

    set(${_sources} ${${_sources}} ${_src_FILE})

  endforeach ()

endmacro()


macro(KDE3_ADD_MOC_FILES _sources)
  foreach (_current_FILE ${ARGN})

    get_filename_component(_tmp_FILE ${_current_FILE} ABSOLUTE)

    get_filename_component(_basename ${_tmp_FILE} NAME_WE)
    set(_moc ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.moc.cpp)

    add_custom_command(OUTPUT ${_moc}
      COMMAND ${QT_MOC_EXECUTABLE}
      ARGS ${_tmp_FILE} -o ${_moc}
      DEPENDS ${_tmp_FILE}
    )

    set(${_sources} ${${_sources}} ${_moc})

  endforeach ()
endmacro()


get_filename_component( KDE3_MODULE_DIR  ${CMAKE_CURRENT_LIST_FILE} PATH)

macro(KDE3_ADD_UI_FILES _sources )
  foreach (_current_FILE ${ARGN})

    get_filename_component(_tmp_FILE ${_current_FILE} ABSOLUTE)

    get_filename_component(_basename ${_tmp_FILE} NAME_WE)
    set(_header ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.h)
    set(_src ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.cpp)
    set(_moc ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.moc.cpp)

    add_custom_command(OUTPUT ${_header}
      COMMAND ${QT_UIC_EXECUTABLE}
      ARGS  -L ${KDE3_LIB_DIR}/kde3/plugins/designer -nounload -o ${_header} ${CMAKE_CURRENT_SOURCE_DIR}/${_current_FILE}
      DEPENDS ${_tmp_FILE}
    )

    add_custom_command(OUTPUT ${_src}
      COMMAND ${CMAKE_COMMAND}
      ARGS
        -DKDE_UIC_PLUGIN_DIR:FILEPATH=${KDE3_LIB_DIR}/kde3/plugins/designer
        -DKDE_UIC_EXECUTABLE:FILEPATH=${QT_UIC_EXECUTABLE}
        -DKDE_UIC_FILE:FILEPATH=${_tmp_FILE}
        -DKDE_UIC_CPP_FILE:FILEPATH=${_src}
        -DKDE_UIC_H_FILE:FILEPATH=${_header}
        -P ${KDE3_MODULE_DIR}/kde3uic.cmake
      DEPENDS ${_header}
    )

    add_custom_command(OUTPUT ${_moc}
      COMMAND ${QT_MOC_EXECUTABLE}
      ARGS ${_header} -o ${_moc}
      DEPENDS ${_header}
    )

    set(${_sources} ${${_sources}} ${_src} ${_moc} )

  endforeach ()
endmacro()


macro(KDE3_AUTOMOC)
  set(_matching_FILES )
  foreach (_current_FILE ${ARGN})

    get_filename_component(_abs_FILE ${_current_FILE} ABSOLUTE)

    # if "SKIP_AUTOMOC" is set to true, we will not handle this file here.
    # here. this is required to make bouic work correctly:
    # we need to add generated .cpp files to the sources (to compile them),
    # but we cannot let automoc handle them, as the .cpp files don't exist yet when
    # cmake is run for the very first time on them -> however the .cpp files might
    # exist at a later run. at that time we need to skip them, so that we don't add two
    # different rules for the same moc file
    get_source_file_property(_skip ${_abs_FILE} SKIP_AUTOMOC)

    if (EXISTS ${_abs_FILE} AND NOT _skip)

      cmake_policy(PUSH)
      cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
      file(STRINGS ${_abs_FILE} _match REGEX "#include +[^ ]+\\.moc[\">]")
      cmake_policy(POP)

      get_filename_component(_abs_PATH ${_abs_FILE} PATH)

      foreach (_current_MOC_INC IN LISTS _match)
        string(REGEX MATCH "[^ <\"]+\\.moc" _current_MOC "${_current_MOC_INC}")

        get_filename_component(_basename ${_current_MOC} NAME_WE)
#       set(_header ${CMAKE_CURRENT_SOURCE_DIR}/${_basename}.h)
        set(_header ${_abs_PATH}/${_basename}.h)
        set(_moc    ${CMAKE_CURRENT_BINARY_DIR}/${_current_MOC})

        add_custom_command(OUTPUT ${_moc}
          COMMAND ${QT_MOC_EXECUTABLE}
          ARGS ${_header} -o ${_moc}
          DEPENDS ${_header}
        )

        set_property(SOURCE "${_abs_FILE}" APPEND PROPERTY OBJECT_DEPENDS "${_moc}")

      endforeach ()
      unset(_match)
      unset(_header)
      unset(_moc)
    endif ()
  endforeach ()
endmacro()

# Only used internally by kde3_install_icons().
macro (_KDE3_ADD_ICON_INSTALL_RULE _install_SCRIPT _install_PATH _group _orig_NAME _install_NAME)

  # if the string doesn't match the pattern, the result is the full string, so all three have the same content
  if (NOT ${_group} STREQUAL ${_install_NAME} )
    set(_icon_GROUP "actions")

    if (${_group} STREQUAL "mime")
      set(_icon_GROUP  "mimetypes")
    endif ()

    if (${_group} STREQUAL "filesys")
      set(_icon_GROUP  "filesystems")
    endif ()

    if (${_group} STREQUAL "device")
      set(_icon_GROUP  "devices")
    endif ()

    if (${_group} STREQUAL "app")
      set(_icon_GROUP  "apps")
    endif ()

    if (${_group} STREQUAL "action")
      set(_icon_GROUP  "actions")
    endif ()

    # message(STATUS "icon: ${_current_ICON} size: ${_size} group: ${_group} name: ${_name}" )
    install(FILES ${_orig_NAME} DESTINATION ${_install_PATH}/${_icon_GROUP}/ RENAME ${_install_NAME} )
  endif ()

endmacro ()


macro (KDE3_INSTALL_ICONS _theme )
  set(_defaultpath "${CMAKE_INSTALL_PREFIX}/share/icons")
  # first the png icons
  file(GLOB _icons *.png)
  foreach (_current_ICON ${_icons} )
    string(REGEX REPLACE "^.*/[a-zA-Z]+([0-9]+)\\-([a-z]+)\\-(.+\\.png)$" "\\1" _size  "${_current_ICON}")
    string(REGEX REPLACE "^.*/[a-zA-Z]+([0-9]+)\\-([a-z]+)\\-(.+\\.png)$" "\\2" _group "${_current_ICON}")
    string(REGEX REPLACE "^.*/[a-zA-Z]+([0-9]+)\\-([a-z]+)\\-(.+\\.png)$" "\\3" _name  "${_current_ICON}")
    _KDE3_ADD_ICON_INSTALL_RULE(${CMAKE_CURRENT_BINARY_DIR}/install_icons.cmake
                                ${_defaultpath}/${_theme}/${_size}x${_size}
                                ${_group} ${_current_ICON} ${_name})
  endforeach ()

  # and now the svg icons
  file(GLOB _icons *.svgz)
  foreach (_current_ICON ${_icons} )
    string(REGEX REPLACE "^.*/crsc\\-([a-z]+)\\-(.+\\.svgz)$" "\\1" _group "${_current_ICON}")
    string(REGEX REPLACE "^.*/crsc\\-([a-z]+)\\-(.+\\.svgz)$" "\\2" _name "${_current_ICON}")
    _KDE3_ADD_ICON_INSTALL_RULE(${CMAKE_CURRENT_BINARY_DIR}/install_icons.cmake
                                ${_defaultpath}/${_theme}/scalable
                                ${_group} ${_current_ICON} ${_name})
  endforeach ()

endmacro ()

macro(KDE3_INSTALL_LIBTOOL_FILE _target)
  get_target_property(_target_location ${_target} LOCATION)

  get_filename_component(_laname ${_target_location} NAME_WE)
  get_filename_component(_soname ${_target_location} NAME)
  set(_laname ${CMAKE_CURRENT_BINARY_DIR}/${_laname}.la)

  file(WRITE ${_laname} "# ${_laname} - a libtool library file, generated by cmake \n")
  file(APPEND ${_laname} "# The name that we can dlopen(3).\n")
  file(APPEND ${_laname} "dlname='${_soname}'\n")
  file(APPEND ${_laname} "# Names of this library\n")
  if(CYGWIN)
    file(APPEND ${_laname} "library_names='${_soname}'\n")
  else()
    file(APPEND ${_laname} "library_names='${_soname} ${_soname} ${_soname}'\n")
  endif()
  file(APPEND ${_laname} "# The name of the static archive\n")
  file(APPEND ${_laname} "old_library=''\n")
  file(APPEND ${_laname} "# Libraries that this one depends upon.\n")
  file(APPEND ${_laname} "dependency_libs=''\n")
#   file(APPEND ${_laname} "dependency_libs='${${_target}_LIB_DEPENDS}'\n")
  file(APPEND ${_laname} "# Version information.\ncurrent=0\nage=0\nrevision=0\n")
  file(APPEND ${_laname} "# Is this an already installed library?\ninstalled=yes\n")
  file(APPEND ${_laname} "# Should we warn about portability when linking against -modules?\nshouldnotlink=yes\n")
  file(APPEND ${_laname} "# Files to dlopen/dlpreopen\ndlopen=''\ndlpreopen=''\n")
  file(APPEND ${_laname} "# Directory that this library needs to be installed in:\n")
  file(APPEND ${_laname} "libdir='${CMAKE_INSTALL_PREFIX}/lib/kde3'\n")

  install_files(${KDE3_LIBTOOL_DIR} FILES ${_laname})
endmacro()


macro(KDE3_CREATE_FINAL_FILE _filename)
  file(WRITE ${_filename} "//autogenerated file\n")
  foreach (_current_FILE ${ARGN})
    file(APPEND ${_filename} "#include \"${_current_FILE}\"\n")
  endforeach ()

endmacro()


# option(KDE3_ENABLE_FINAL "Enable final all-in-one compilation")
option(KDE3_BUILD_TESTS  "Build the tests")


macro(KDE3_ADD_KPART _target_NAME _with_PREFIX)
#is the first argument is "WITH_PREFIX" then keep the standard "lib" prefix, otherwise SET the prefix empty
  if (${_with_PREFIX} STREQUAL "WITH_PREFIX")
    set(_first_SRC)
  else ()
    set(_first_SRC ${_with_PREFIX})
  endif ()

#    if (KDE3_ENABLE_FINAL)
#       kde3_create_final_file(${_target_NAME}_final.cpp ${_first_SRC} ${ARGN})
#       add_library(${_target_NAME} MODULE  ${_target_NAME}_final.cpp)
#    else ()
  add_library(${_target_NAME} MODULE ${_first_SRC} ${ARGN})
#    endif ()

  if(_first_SRC)
    set_target_properties(${_target_NAME} PROPERTIES PREFIX "")
  endif()

  kde3_install_libtool_file(${_target_NAME})
endmacro()


macro(KDE3_ADD_KDEINIT_EXECUTABLE _target_NAME )

#    if (KDE3_ENABLE_FINAL)
#       kde3_create_final_file(${_target_NAME}_final.cpp ${ARGN})
#       add_library(kdeinit_${_target_NAME} SHARED  ${_target_NAME}_final.cpp)
#    else ()
  add_library(kdeinit_${_target_NAME} SHARED ${ARGN} )
#    endif ()

  configure_file(${KDE3_MODULE_DIR}/kde3init_dummy.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/${_target_NAME}_dummy.cpp)

  add_executable( ${_target_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${_target_NAME}_dummy.cpp )
  target_link_libraries( ${_target_NAME} kdeinit_${_target_NAME} )

endmacro()


macro(KDE3_ADD_EXECUTABLE _target_NAME )

#    if (KDE3_ENABLE_FINAL)
#       kde3_create_final_file(${_target_NAME}_final.cpp ${ARGN})
#       add_executable(${_target_NAME} ${_target_NAME}_final.cpp)
#    else ()
  add_executable(${_target_NAME} ${ARGN} )
#    endif ()

endmacro()
