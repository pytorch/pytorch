# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
Qt4Macros
---------



This file is included by FindQt4.cmake, don't include it directly.
#]=======================================================================]

######################################
#
#       Macros for building Qt files
#
######################################


macro (QT4_EXTRACT_OPTIONS _qt4_files _qt4_options _qt4_target)
  set(${_qt4_files})
  set(${_qt4_options})
  set(_QT4_DOING_OPTIONS FALSE)
  set(_QT4_DOING_TARGET FALSE)
  foreach(_currentArg ${ARGN})
    if ("x${_currentArg}" STREQUAL "xOPTIONS")
      set(_QT4_DOING_OPTIONS TRUE)
    elseif ("x${_currentArg}" STREQUAL "xTARGET")
      set(_QT4_DOING_TARGET TRUE)
    else ()
      if(_QT4_DOING_TARGET)
        set(${_qt4_target} "${_currentArg}")
      elseif(_QT4_DOING_OPTIONS)
        list(APPEND ${_qt4_options} "${_currentArg}")
      else()
        list(APPEND ${_qt4_files} "${_currentArg}")
      endif()
    endif ()
  endforeach()
endmacro ()


# macro used to create the names of output files preserving relative dirs
macro (QT4_MAKE_OUTPUT_FILE infile prefix ext outfile )
  string(LENGTH ${CMAKE_CURRENT_BINARY_DIR} _binlength)
  string(LENGTH ${infile} _infileLength)
  set(_checkinfile ${CMAKE_CURRENT_SOURCE_DIR})
  if(_infileLength GREATER _binlength)
    string(SUBSTRING "${infile}" 0 ${_binlength} _checkinfile)
    if(_checkinfile STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
      file(RELATIVE_PATH rel ${CMAKE_CURRENT_BINARY_DIR} ${infile})
    else()
      file(RELATIVE_PATH rel ${CMAKE_CURRENT_SOURCE_DIR} ${infile})
    endif()
  else()
    file(RELATIVE_PATH rel ${CMAKE_CURRENT_SOURCE_DIR} ${infile})
  endif()
  if(WIN32 AND rel MATCHES "^([a-zA-Z]):(.*)$") # absolute path
    set(rel "${CMAKE_MATCH_1}_${CMAKE_MATCH_2}")
  endif()
  set(_outfile "${CMAKE_CURRENT_BINARY_DIR}/${rel}")
  string(REPLACE ".." "__" _outfile ${_outfile})
  get_filename_component(outpath ${_outfile} PATH)
  get_filename_component(_outfile ${_outfile} NAME_WE)
  file(MAKE_DIRECTORY ${outpath})
  set(${outfile} ${outpath}/${prefix}${_outfile}.${ext})
endmacro ()


macro (QT4_GET_MOC_FLAGS _moc_flags)
  set(${_moc_flags})
  get_directory_property(_inc_DIRS INCLUDE_DIRECTORIES)

  foreach(_current ${_inc_DIRS})
    if("${_current}" MATCHES "\\.framework/?$")
      string(REGEX REPLACE "/[^/]+\\.framework" "" framework_path "${_current}")
      set(${_moc_flags} ${${_moc_flags}} "-F${framework_path}")
    else()
      set(${_moc_flags} ${${_moc_flags}} "-I${_current}")
    endif()
  endforeach()

  get_directory_property(_defines COMPILE_DEFINITIONS)
  foreach(_current ${_defines})
    set(${_moc_flags} ${${_moc_flags}} "-D${_current}")
  endforeach()

  if(Q_WS_WIN)
    set(${_moc_flags} ${${_moc_flags}} -DWIN32)
  endif()

endmacro()


# helper macro to set up a moc rule
function (QT4_CREATE_MOC_COMMAND infile outfile moc_flags moc_options moc_target)
  # For Windows, create a parameters file to work around command line length limit
  # Pass the parameters in a file.  Set the working directory to
  # be that containing the parameters file and reference it by
  # just the file name.  This is necessary because the moc tool on
  # MinGW builds does not seem to handle spaces in the path to the
  # file given with the @ syntax.
  get_filename_component(_moc_outfile_name "${outfile}" NAME)
  get_filename_component(_moc_outfile_dir "${outfile}" PATH)
  if(_moc_outfile_dir)
    set(_moc_working_dir WORKING_DIRECTORY ${_moc_outfile_dir})
  endif()
  set (_moc_parameters_file ${outfile}_parameters)
  set (_moc_parameters ${moc_flags} ${moc_options} -o "${outfile}" "${infile}")
  string (REPLACE ";" "\n" _moc_parameters "${_moc_parameters}")

  if(moc_target)
    set (_moc_parameters_file ${_moc_parameters_file}$<$<BOOL:$<CONFIGURATION>>:_$<CONFIGURATION>>)
    set(targetincludes "$<TARGET_PROPERTY:${moc_target},INCLUDE_DIRECTORIES>")
    set(targetdefines "$<TARGET_PROPERTY:${moc_target},COMPILE_DEFINITIONS>")

    set(targetincludes "$<$<BOOL:${targetincludes}>:-I$<JOIN:${targetincludes},\n-I>\n>")
    set(targetdefines "$<$<BOOL:${targetdefines}>:-D$<JOIN:${targetdefines},\n-D>\n>")

    file (GENERATE
      OUTPUT ${_moc_parameters_file}
      CONTENT "${targetdefines}${targetincludes}${_moc_parameters}\n"
    )

    set(targetincludes)
    set(targetdefines)
  else()
    file(
      CONFIGURE
      OUTPUT "${_moc_parameters_file}"
      CONTENT "${_moc_parameters}\n"
      @ONLY
    )
  endif()

  set(_moc_extra_parameters_file @${_moc_parameters_file})
  add_custom_command(OUTPUT ${outfile}
                      COMMAND Qt4::moc ${_moc_extra_parameters_file}
                      DEPENDS ${infile} ${_moc_parameters_file}
                      ${_moc_working_dir}
                      VERBATIM)
endfunction ()


macro (QT4_GENERATE_MOC infile outfile )
  # get include dirs and flags
  qt4_get_moc_flags(moc_flags)
  get_filename_component(abs_infile ${infile} ABSOLUTE)
  set(_outfile "${outfile}")
  if(NOT IS_ABSOLUTE "${outfile}")
    set(_outfile "${CMAKE_CURRENT_BINARY_DIR}/${outfile}")
  endif()

  if (${ARGC} GREATER 3 AND "x${ARGV2}" STREQUAL "xTARGET")
    set(moc_target ${ARGV3})
  endif()
  qt4_create_moc_command(${abs_infile} ${_outfile} "${moc_flags}" "" "${moc_target}")
  set_property(SOURCE ${outfile} PROPERTY SKIP_AUTOMOC TRUE)  # don't run automoc on this file
  set_property(SOURCE ${outfile} PROPERTY SKIP_AUTOUIC TRUE)  # don't run autouic on this file
endmacro ()


# qt4_wrap_cpp(outfiles inputfile ... )
macro (QT4_WRAP_CPP outfiles )
  # get include dirs
  qt4_get_moc_flags(moc_flags)
  qt4_extract_options(moc_files moc_options moc_target ${ARGN})

  foreach (it ${moc_files})
    get_filename_component(it ${it} ABSOLUTE)
    qt4_make_output_file(${it} moc_ cxx outfile)
    qt4_create_moc_command(${it} ${outfile} "${moc_flags}" "${moc_options}" "${moc_target}")
    set_property(SOURCE ${outfile} PROPERTY SKIP_AUTOMOC TRUE)  # don't run automoc on this file
    set_property(SOURCE ${outfile} PROPERTY SKIP_AUTOUIC TRUE)  # don't run autouic on this file
    set(${outfiles} ${${outfiles}} ${outfile})
  endforeach()

endmacro ()


# qt4_wrap_ui(outfiles inputfile ... )
macro (QT4_WRAP_UI outfiles )
  qt4_extract_options(ui_files ui_options ui_target ${ARGN})

  foreach (it ${ui_files})
    get_filename_component(outfile ${it} NAME_WE)
    get_filename_component(infile ${it} ABSOLUTE)
    set(outfile ${CMAKE_CURRENT_BINARY_DIR}/ui_${outfile}.h)
    add_custom_command(OUTPUT ${outfile}
      COMMAND Qt4::uic
      ARGS ${ui_options} -o ${outfile} ${infile}
      MAIN_DEPENDENCY ${infile} VERBATIM)
    set_property(SOURCE ${outfile} PROPERTY SKIP_AUTOMOC TRUE)  # don't run automoc on this file
    set_property(SOURCE ${outfile} PROPERTY SKIP_AUTOUIC TRUE)  # don't run autouic on this file
    set(${outfiles} ${${outfiles}} ${outfile})
  endforeach ()

endmacro ()


# qt4_add_resources(outfiles inputfile ... )
macro (QT4_ADD_RESOURCES outfiles )
  qt4_extract_options(rcc_files rcc_options rcc_target ${ARGN})

  foreach (it ${rcc_files})
    get_filename_component(outfilename ${it} NAME_WE)
    get_filename_component(infile ${it} ABSOLUTE)
    get_filename_component(rc_path ${infile} PATH)
    set(outfile ${CMAKE_CURRENT_BINARY_DIR}/qrc_${outfilename}.cxx)

    set(_RC_DEPENDS)
    if(EXISTS "${infile}")
      #  parse file for dependencies
      #  all files are absolute paths or relative to the location of the qrc file
      file(READ "${infile}" _RC_FILE_CONTENTS)
      string(REGEX MATCHALL "<file[^<]+" _RC_FILES "${_RC_FILE_CONTENTS}")
      foreach(_RC_FILE ${_RC_FILES})
        string(REGEX REPLACE "^<file[^>]*>" "" _RC_FILE "${_RC_FILE}")
        if(NOT IS_ABSOLUTE "${_RC_FILE}")
          set(_RC_FILE "${rc_path}/${_RC_FILE}")
        endif()
        set(_RC_DEPENDS ${_RC_DEPENDS} "${_RC_FILE}")
      endforeach()
      unset(_RC_FILES)
      unset(_RC_FILE_CONTENTS)
      # Since this cmake macro is doing the dependency scanning for these files,
      # let's make a configured file and add it as a dependency so cmake is run
      # again when dependencies need to be recomputed.
      qt4_make_output_file("${infile}" "" "qrc.depends" out_depends)
      configure_file("${infile}" "${out_depends}" COPYONLY)
    else()
      # The .qrc file does not exist (yet). Let's add a dependency and hope
      # that it will be generated later
      set(out_depends)
    endif()

    add_custom_command(OUTPUT ${outfile}
      COMMAND Qt4::rcc
      ARGS ${rcc_options} -name ${outfilename} -o ${outfile} ${infile}
      MAIN_DEPENDENCY ${infile}
      DEPENDS ${_RC_DEPENDS} "${out_depends}" VERBATIM)
    set_property(SOURCE ${outfile} PROPERTY SKIP_AUTOMOC TRUE)  # don't run automoc on this file
    set_property(SOURCE ${outfile} PROPERTY SKIP_AUTOUIC TRUE)  # don't run autouic on this file
    set(${outfiles} ${${outfiles}} ${outfile})
  endforeach ()

endmacro ()


macro(QT4_ADD_DBUS_INTERFACE _sources _interface _basename)
  get_filename_component(_infile ${_interface} ABSOLUTE)
  set(_header "${CMAKE_CURRENT_BINARY_DIR}/${_basename}.h")
  set(_impl   "${CMAKE_CURRENT_BINARY_DIR}/${_basename}.cpp")
  set(_moc    "${CMAKE_CURRENT_BINARY_DIR}/${_basename}.moc")

  get_property(_nonamespace SOURCE ${_interface} PROPERTY NO_NAMESPACE)
  if(_nonamespace)
    set(_params -N -m)
  else()
    set(_params -m)
  endif()

  get_property(_classname SOURCE ${_interface} PROPERTY CLASSNAME)
  if(_classname)
    set(_params ${_params} -c ${_classname})
  endif()

  get_property(_include SOURCE ${_interface} PROPERTY INCLUDE)
  if(_include)
    set(_params ${_params} -i ${_include})
  endif()

  add_custom_command(OUTPUT "${_impl}" "${_header}"
      COMMAND Qt4::qdbusxml2cpp ${_params} -p ${_basename} ${_infile}
      DEPENDS ${_infile} VERBATIM)

  set_property(SOURCE ${_impl} PROPERTY SKIP_AUTOMOC TRUE)  # don't run automoc on this file
  set_property(SOURCE ${_impl} PROPERTY SKIP_AUTOUIC TRUE)  # don't run autouic on this file

  qt4_generate_moc("${_header}" "${_moc}")

  list(APPEND ${_sources} "${_impl}" "${_header}" "${_moc}")
  set_property(SOURCE "${_impl}" APPEND PROPERTY OBJECT_DEPENDS "${_moc}")

endmacro()


macro(QT4_ADD_DBUS_INTERFACES _sources)
  foreach (_current_FILE ${ARGN})
    get_filename_component(_infile ${_current_FILE} ABSOLUTE)
    get_filename_component(_basename ${_current_FILE} NAME)
    # get the part before the ".xml" suffix
    string(TOLOWER ${_basename} _basename)
    string(REGEX REPLACE "(.*\\.)?([^\\.]+)\\.xml" "\\2" _basename ${_basename})
    qt4_add_dbus_interface(${_sources} ${_infile} ${_basename}interface)
  endforeach ()
endmacro()


macro(QT4_GENERATE_DBUS_INTERFACE _header) # _customName OPTIONS -some -options )
  qt4_extract_options(_customName _qt4_dbus_options _qt4_dbus_target ${ARGN})

  get_filename_component(_in_file ${_header} ABSOLUTE)
  get_filename_component(_basename ${_header} NAME_WE)

  if (_customName)
    if (IS_ABSOLUTE ${_customName})
      get_filename_component(_containingDir ${_customName} PATH)
      if (NOT EXISTS ${_containingDir})
        file(MAKE_DIRECTORY "${_containingDir}")
      endif()
      set(_target ${_customName})
    else()
      set(_target ${CMAKE_CURRENT_BINARY_DIR}/${_customName})
    endif()
  else ()
    set(_target ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.xml)
  endif ()

  add_custom_command(OUTPUT ${_target}
      COMMAND Qt4::qdbuscpp2xml ${_qt4_dbus_options} ${_in_file} -o ${_target}
      DEPENDS ${_in_file} VERBATIM
  )
endmacro()


macro(QT4_ADD_DBUS_ADAPTOR _sources _xml_file _include _parentClass) # _optionalBasename _optionalClassName)
  get_filename_component(_infile ${_xml_file} ABSOLUTE)

  unset(_optionalBasename)
  if(${ARGC} GREATER 4)
    set(_optionalBasename "${ARGV4}")
  endif()
  if (_optionalBasename)
    set(_basename ${_optionalBasename} )
  else ()
    string(REGEX REPLACE "(.*[/\\.])?([^\\.]+)\\.xml" "\\2adaptor" _basename ${_infile})
    string(TOLOWER ${_basename} _basename)
  endif ()

  unset(_optionalClassName)
  if(${ARGC} GREATER 5)
    set(_optionalClassName "${ARGV5}")
  endif()
  set(_header "${CMAKE_CURRENT_BINARY_DIR}/${_basename}.h")
  set(_impl   "${CMAKE_CURRENT_BINARY_DIR}/${_basename}.cpp")
  set(_moc    "${CMAKE_CURRENT_BINARY_DIR}/${_basename}.moc")

  if(_optionalClassName)
    add_custom_command(OUTPUT "${_impl}" "${_header}"
       COMMAND Qt4::qdbusxml2cpp -m -a ${_basename} -c ${_optionalClassName} -i ${_include} -l ${_parentClass} ${_infile}
       DEPENDS ${_infile} VERBATIM
    )
  else()
    add_custom_command(OUTPUT "${_impl}" "${_header}"
       COMMAND Qt4::qdbusxml2cpp -m -a ${_basename} -i ${_include} -l ${_parentClass} ${_infile}
       DEPENDS ${_infile} VERBATIM
     )
  endif()

  qt4_generate_moc("${_header}" "${_moc}")
  set_property(SOURCE ${_impl} PROPERTY SKIP_AUTOMOC TRUE)  # don't run automoc on this file
  set_property(SOURCE ${_impl} PROPERTY SKIP_AUTOUIC TRUE)  # don't run autouic on this file
  set_property(SOURCE "${_impl}" APPEND PROPERTY OBJECT_DEPENDS "${_moc}")

  list(APPEND ${_sources} "${_impl}" "${_header}" "${_moc}")
endmacro()


macro(QT4_AUTOMOC)
  if(NOT CMAKE_MINIMUM_REQUIRED_VERSION VERSION_LESS 2.8.11)
    message(DEPRECATION "The qt4_automoc macro is obsolete. Use the CMAKE_AUTOMOC feature instead.")
  endif()
  qt4_get_moc_flags(_moc_INCS)

  set(_matching_FILES )
  foreach (_current_FILE ${ARGN})

    get_filename_component(_abs_FILE ${_current_FILE} ABSOLUTE)
    # if "SKIP_AUTOMOC" is set to true, we will not handle this file here.
    # This is required to make uic work correctly:
    # we need to add generated .cpp files to the sources (to compile them),
    # but we cannot let automoc handle them, as the .cpp files don't exist yet when
    # cmake is run for the very first time on them -> however the .cpp files might
    # exist at a later run. at that time we need to skip them, so that we don't add two
    # different rules for the same moc file
    get_property(_skip SOURCE ${_abs_FILE} PROPERTY SKIP_AUTOMOC)

    if ( NOT _skip AND EXISTS ${_abs_FILE} )

      file(READ ${_abs_FILE} _contents)

      get_filename_component(_abs_PATH ${_abs_FILE} PATH)

      string(REGEX MATCHALL "# *include +[^ ]+\\.moc[\">]" _match "${_contents}")
      if(_match)
        foreach (_current_MOC_INC ${_match})
          string(REGEX MATCH "[^ <\"]+\\.moc" _current_MOC "${_current_MOC_INC}")

          get_filename_component(_basename ${_current_MOC} NAME_WE)
          if(EXISTS ${_abs_PATH}/${_basename}.hpp)
            set(_header ${_abs_PATH}/${_basename}.hpp)
          else()
            set(_header ${_abs_PATH}/${_basename}.h)
          endif()
          set(_moc    ${CMAKE_CURRENT_BINARY_DIR}/${_current_MOC})
          qt4_create_moc_command(${_header} ${_moc} "${_moc_INCS}" "" "")
          set_property(SOURCE "${_abs_FILE}" APPEND PROPERTY OBJECT_DEPENDS "${_moc}")
        endforeach ()
      endif()
    endif ()
  endforeach ()
endmacro()


macro(QT4_CREATE_TRANSLATION _qm_files)
  qt4_extract_options(_lupdate_files _lupdate_options _lupdate_target ${ARGN})
  set(_my_sources)
  set(_my_dirs)
  set(_my_tsfiles)
  set(_ts_pro)
  foreach (_file ${_lupdate_files})
    get_filename_component(_ext ${_file} EXT)
    get_filename_component(_abs_FILE ${_file} ABSOLUTE)
    if(_ext MATCHES "ts")
      list(APPEND _my_tsfiles ${_abs_FILE})
    else()
      if(NOT _ext)
        list(APPEND _my_dirs ${_abs_FILE})
      else()
        list(APPEND _my_sources ${_abs_FILE})
      endif()
    endif()
  endforeach()
  foreach(_ts_file ${_my_tsfiles})
    if(_my_sources)
      # make a .pro file to call lupdate on, so we don't make our commands too
      # long for some systems
      get_filename_component(_ts_name ${_ts_file} NAME)
      set(_ts_pro ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${_ts_name}_lupdate.pro)
      set(_pro_srcs)
      foreach(_pro_src ${_my_sources})
        string(APPEND _pro_srcs " \\\n  \"${_pro_src}\"")
      endforeach()
      set(_pro_includes)
      get_directory_property(_inc_DIRS INCLUDE_DIRECTORIES)
      list(REMOVE_DUPLICATES _inc_DIRS)
      foreach(_pro_include ${_inc_DIRS})
        get_filename_component(_abs_include "${_pro_include}" ABSOLUTE)
        string(APPEND _pro_includes " \\\n  \"${_abs_include}\"")
      endforeach()
      file(GENERATE OUTPUT ${_ts_pro} CONTENT "SOURCES =${_pro_srcs}\nINCLUDEPATH =${_pro_includes}\n")
    endif()
    add_custom_command(OUTPUT ${_ts_file}
        COMMAND Qt4::lupdate
        ARGS ${_lupdate_options} ${_ts_pro} ${_my_dirs} -ts ${_ts_file}
        DEPENDS ${_my_sources} ${_ts_pro} VERBATIM)
  endforeach()
  qt4_add_translation(${_qm_files} ${_my_tsfiles})
endmacro()


macro(QT4_ADD_TRANSLATION _qm_files)
  foreach (_current_FILE ${ARGN})
    get_filename_component(_abs_FILE ${_current_FILE} ABSOLUTE)
    get_filename_component(qm ${_abs_FILE} NAME)
    # everything before the last dot has to be considered the file name (including other dots)
    string(REGEX REPLACE "\\.[^.]*$" "" FILE_NAME ${qm})
    get_source_file_property(output_location ${_abs_FILE} OUTPUT_LOCATION)
    if(output_location)
      file(MAKE_DIRECTORY "${output_location}")
      set(qm "${output_location}/${FILE_NAME}.qm")
    else()
      set(qm "${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.qm")
    endif()

    add_custom_command(OUTPUT ${qm}
       COMMAND Qt4::lrelease
       ARGS ${_abs_FILE} -qm ${qm}
       DEPENDS ${_abs_FILE} VERBATIM
    )
    set(${_qm_files} ${${_qm_files}} ${qm})
  endforeach ()
endmacro()

function(qt4_use_modules _target _link_type)
  if(NOT CMAKE_MINIMUM_REQUIRED_VERSION VERSION_LESS 2.8.11)
    message(DEPRECATION "The qt4_use_modules function is obsolete. Use target_link_libraries with IMPORTED targets instead.")
  endif()
  if ("${_link_type}" STREQUAL "LINK_PUBLIC" OR "${_link_type}" STREQUAL "LINK_PRIVATE")
    set(modules ${ARGN})
    set(link_type ${_link_type})
  else()
    set(modules ${_link_type} ${ARGN})
  endif()
  foreach(_module ${modules})
    string(TOUPPER ${_module} _ucmodule)
    set(_targetPrefix QT_QT${_ucmodule})
    if (_ucmodule STREQUAL QAXCONTAINER OR _ucmodule STREQUAL QAXSERVER)
      if (NOT QT_Q${_ucmodule}_FOUND)
        message(FATAL_ERROR "Can not use \"${_module}\" module which has not yet been found.")
      endif()
      set(_targetPrefix QT_Q${_ucmodule})
    else()
      if (NOT QT_QT${_ucmodule}_FOUND)
        message(FATAL_ERROR "Can not use \"${_module}\" module which has not yet been found.")
      endif()
      if ("${_ucmodule}" STREQUAL "MAIN")
        message(FATAL_ERROR "Can not use \"${_module}\" module with qt4_use_modules.")
      endif()
    endif()
    target_link_libraries(${_target} ${link_type} ${${_targetPrefix}_LIBRARIES})
    set_property(TARGET ${_target} APPEND PROPERTY INCLUDE_DIRECTORIES ${${_targetPrefix}_INCLUDE_DIR} ${QT_HEADERS_DIR} ${QT_MKSPECS_DIR}/default)
    set_property(TARGET ${_target} APPEND PROPERTY COMPILE_DEFINITIONS ${${_targetPrefix}_COMPILE_DEFINITIONS})
  endforeach()
endfunction()
