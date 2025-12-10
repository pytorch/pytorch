# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
DeployQt4
---------

.. note::

  This module is for Qt version 4.  New code should follow the
  :manual:`cmake-qt(7)` instead of using this module.

This module provides a collection of CMake utility commands useful for
assembling and deploying standalone Qt4 executables.

Load this module in a CMake project with:

.. code-block:: cmake

  include(DeployQt4)

Commands
^^^^^^^^

This module provides the following commands:

* :command:`write_qt4_conf`
* :command:`resolve_qt4_paths`
* :command:`fixup_qt4_executable`
* :command:`install_qt4_plugin_path`
* :command:`install_qt4_plugin`
* :command:`install_qt4_executable`

.. command:: write_qt4_conf

  Creates a Qt configuration file:

  .. code-block:: cmake

    write_qt4_conf(<qt-conf-dir> <qt-conf-contents>)

  This command writes a ``qt.conf`` file with the ``<qt-conf-contents>`` into
  the ``<qt-conf-dir>`` directory.

.. command:: resolve_qt4_paths

  Resolves relative paths to absolute:

  .. code-block:: cmake

    resolve_qt4_paths(<paths-var> [<executable-path>])

  This command loops through the ``<paths-var>`` list and if any path doesn't
  exist, it resolves them relative to the ``<executable-path>`` (if supplied)
  or the :variable:`CMAKE_INSTALL_PREFIX`.

.. command:: fixup_qt4_executable

  Fixes up a Qt4 executable:

  .. code-block:: cmake

    fixup_qt4_executable(
      <executable>
      [<qtplugins> <libs> <dirs> <plugins-dir> <request-qt-conf>]
    )

  This command copies Qt plugins, writes a Qt configuration file (if needed)
  and fixes up a Qt4 executable using the :module:`BundleUtilities` module
  so it is standalone and can be drag-and-drop copied to another machine as
  long as all of the system libraries are compatible.

  ``<executable>`` should point to the executable to be fixed-up.

  ``<qtplugins>`` should contain a list of the names or paths of any Qt plugins
  to be installed.

  ``<libs>`` will be passed to the :module:`BundleUtilities` module and should
  be a list of any already installed plugins, libraries or executables to also
  be fixed-up.

  ``<dirs>`` will be passed to the :module:`BundleUtilities` module and should
  contain directories to be searched to find library dependencies.

  ``<plugins-dir>`` allows a custom plugins directory to be used.

  ``<request-qt-conf>`` will force a ``qt.conf`` file to be written even if not
  needed.

.. command:: install_qt4_plugin_path

  Installs a resolved Qt4 plugin:

  .. code-block:: cmake

    install_qt4_plugin_path(
      <plugin>
      <executable>
      <copy>
      <installed-plugin-path-var>
      [<plugins-dir> <component> <configurations>]
    )

  This command installs (or copies) a resolved ``<plugin>`` to the default
  plugins directory (or ``<plugins-dir>``) relative to ``<executable>`` and
  stores the result in a variable ``<installed-plugin-path-var>``.

  If ``<copy>`` is set to ``TRUE`` then the plugins will be copied rather than
  installed.  This is to allow this module to be used at CMake time rather than
  install time.

  If ``<component>`` is set then anything installed will use this COMPONENT.

.. command:: install_qt4_plugin

  Installs an unresolved Qt4 plugin:

  .. code-block:: cmake

    install_qt4_plugin(
      <plugin>
      <executable>
      <copy>
      <installed-plugin-path-var>
      [<plugins-dir> <component>]
    )

  This command installs (or copies) an unresolved ``<plugin>`` to the default
  plugins directory (or ``<plugins-dir>``) relative to ``<executable>`` and
  stores the result in a variable ``<installed-plugin-path-var>``.  For other
  arguments, see also :command:`install_qt4_plugin_path`.

.. command:: install_qt4_executable

  Installs Qt plugins, writes a Qt configuration file (if needed) and fixes
  up a Qt4 executable:

  .. code-block:: cmake

    install_qt4_executable(
      <executable>
      [<qtplugins> <libs> <dirs> <plugins-dir> <request-qt-conf> <component>]
    )

  This command uses the :module:`BundleUtilities` module so executable is
  standalone and can be drag-and-drop copied to another machine as long as
  all of the system libraries are compatible.  The executable will be fixed-up
  at install time.  The ``<component>`` is the COMPONENT used for bundle fixup
  and plugin installation.  For other arguments, see also
  :command:`fixup_qt4_executable`.
#]=======================================================================]

# The functions defined in this file depend on the fixup_bundle function
# (and others) found in BundleUtilities.cmake

set(DeployQt4_apple_plugins_dir "PlugIns")

function(write_qt4_conf qt_conf_dir qt_conf_contents)
  set(qt_conf_path "${qt_conf_dir}/qt.conf")
  message(STATUS "Writing ${qt_conf_path}")
  file(WRITE "${qt_conf_path}" "${qt_conf_contents}")
endfunction()

function(resolve_qt4_paths paths_var)
  unset(executable_path)
  if(ARGC GREATER 1)
    set(executable_path ${ARGV1})
  endif()

  set(paths_resolved)
  foreach(path ${${paths_var}})
    if(EXISTS "${path}")
      list(APPEND paths_resolved "${path}")
    else()
      if(${executable_path})
        list(APPEND paths_resolved "${executable_path}/${path}")
      else()
        list(APPEND paths_resolved "\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${path}")
      endif()
    endif()
  endforeach()
  set(${paths_var} ${paths_resolved} PARENT_SCOPE)
endfunction()

cmake_policy(GET CMP0080 _cmp0080_value)
if(NOT DEFINED CMAKE_GENERATOR OR NOT _cmp0080_value STREQUAL "NEW")
  set(_CMP0080_SUPPRESS_WARNING TRUE)
  include("${CMAKE_CURRENT_LIST_DIR}/BundleUtilities.cmake")
  unset(_CMP0080_SUPPRESS_WARNING)

  function(fixup_qt4_executable executable)
    cmake_policy(GET CMP0080 _cmp0080_value)
    if(_cmp0080_value STREQUAL "" AND DEFINED CMAKE_GENERATOR)
      _warn_cmp0080()
    endif()

    unset(qtplugins)
    if(ARGC GREATER 1)
      set(qtplugins ${ARGV1})
    endif()
    unset(libs)
    if(ARGC GREATER 2)
      set(libs ${ARGV2})
    endif()
    unset(dirs)
    if(ARGC GREATER 3)
      set(dirs ${ARGV3})
    endif()
    unset(plugins_dir)
    if(ARGC GREATER 4)
      set(plugins_dir ${ARGV4})
    endif()
    unset(request_qt_conf)
    if(ARGC GREATER 5)
      set(request_qt_conf ${ARGV5})
    endif()

    message(STATUS "fixup_qt4_executable")
    message(STATUS "  executable='${executable}'")
    message(STATUS "  qtplugins='${qtplugins}'")
    message(STATUS "  libs='${libs}'")
    message(STATUS "  dirs='${dirs}'")
    message(STATUS "  plugins_dir='${plugins_dir}'")
    message(STATUS "  request_qt_conf='${request_qt_conf}'")

    if(QT_LIBRARY_DIR)
      list(APPEND dirs "${QT_LIBRARY_DIR}")
    endif()
    if(QT_BINARY_DIR)
      list(APPEND dirs "${QT_BINARY_DIR}")
    endif()

    if(APPLE)
      set(qt_conf_dir "${executable}/Contents/Resources")
      set(executable_path "${executable}")
      set(write_qt_conf TRUE)
      if(NOT DEFINED plugins_dir)
        set(plugins_dir "${DeployQt4_apple_plugins_dir}")
      endif()
    else()
      get_filename_component(executable_path "${executable}" PATH)
      if(NOT executable_path)
        set(executable_path ".")
      endif()
      set(qt_conf_dir "${executable_path}")
      set(write_qt_conf ${request_qt_conf})
    endif()

    foreach(plugin ${qtplugins})
      set(installed_plugin_path "")
      install_qt4_plugin("${plugin}" "${executable}" 1 installed_plugin_path)
      list(APPEND libs ${installed_plugin_path})
    endforeach()

    foreach(lib ${libs})
      if(NOT EXISTS "${lib}")
        message(FATAL_ERROR "Library does not exist: ${lib}")
      endif()
    endforeach()

    resolve_qt4_paths(libs "${executable_path}")

    if(write_qt_conf)
      set(qt_conf_contents "[Paths]\nPlugins = ${plugins_dir}")
      write_qt4_conf("${qt_conf_dir}" "${qt_conf_contents}")
    endif()

    fixup_bundle("${executable}" "${libs}" "${dirs}")
  endfunction()
endif()

function(install_qt4_plugin_path plugin executable copy installed_plugin_path_var)
  unset(plugins_dir)
  if(ARGC GREATER 4)
    set(plugins_dir ${ARGV4})
  endif()
  unset(component)
  if(ARGC GREATER 5)
    set(component ${ARGV5})
  endif()
  unset(configurations)
  if(ARGC GREATER 6)
    set(configurations ${ARGV6})
  endif()

  if(EXISTS "${plugin}")
    if(APPLE)
      if(NOT plugins_dir)
        set(plugins_dir "${DeployQt4_apple_plugins_dir}")
      endif()
      set(plugins_path "${executable}/Contents/${plugins_dir}")
    else()
      get_filename_component(plugins_path "${executable}" PATH)
      if(NOT plugins_path)
        set(plugins_path ".")
      endif()
      if(plugins_dir)
        string(APPEND plugins_path "/${plugins_dir}")
      endif()
    endif()

    set(plugin_group "")

    get_filename_component(plugin_path "${plugin}" PATH)
    get_filename_component(plugin_parent_path "${plugin_path}" PATH)
    get_filename_component(plugin_parent_dir_name "${plugin_parent_path}" NAME)
    get_filename_component(plugin_name "${plugin}" NAME)
    string(TOLOWER "${plugin_parent_dir_name}" plugin_parent_dir_name)

    if("${plugin_parent_dir_name}" STREQUAL "plugins")
      get_filename_component(plugin_group "${plugin_path}" NAME)
      set(${plugin_group_var} "${plugin_group}")
    endif()
    string(APPEND plugins_path "/${plugin_group}")

    if(${copy})
      file(MAKE_DIRECTORY "${plugins_path}")
      file(COPY "${plugin}" DESTINATION "${plugins_path}")
    else()
      get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
      if(configurations AND (_isMultiConfig OR CMAKE_BUILD_TYPE))
        set(configurations CONFIGURATIONS ${configurations})
      else()
        unset(configurations)
      endif()
      install(FILES "${plugin}" DESTINATION "${plugins_path}" ${configurations} ${component})
    endif()
    set(${installed_plugin_path_var} "${plugins_path}/${plugin_name}" PARENT_SCOPE)
  endif()
endfunction()

function(install_qt4_plugin plugin executable copy installed_plugin_path_var)
  unset(plugins_dir)
  if(ARGC GREATER 4)
    set(plugins_dir ${ARGV4})
  endif()
  unset(component)
  if(ARGC GREATER 5)
    set(component ${ARGV5})
  endif()

  if(EXISTS "${plugin}")
    install_qt4_plugin_path("${plugin}" "${executable}" "${copy}" "${installed_plugin_path_var}" "${plugins_dir}" "${component}")
  else()
    string(TOUPPER "QT_${plugin}_PLUGIN" plugin_var)
    set(plugin_release_var "${plugin_var}_RELEASE")
    set(plugin_debug_var "${plugin_var}_DEBUG")
    set(plugin_release "${${plugin_release_var}}")
    set(plugin_debug "${${plugin_debug_var}}")
    if(DEFINED "${plugin_release_var}" AND DEFINED "${plugin_debug_var}" AND NOT EXISTS "${plugin_release}" AND NOT EXISTS "${plugin_debug}")
      message(WARNING "Qt plugin \"${plugin}\" not recognized or found.")
    endif()
    if(NOT EXISTS "${${plugin_debug_var}}")
      set(plugin_debug "${plugin_release}")
    endif()

    get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_isMultiConfig OR CMAKE_BUILD_TYPE)
      set(_RELEASE_CONFIGS ${CMAKE_CONFIGURATION_TYPES} "${CMAKE_BUILD_TYPE}")
      if (_RELEASE_CONFIGS)
        list(FILTER _RELEASE_CONFIGS EXCLUDE REGEX "[Dd][Ee][Bb][Uu][Gg]")
      endif()
      string(REPLACE ";" "|" _RELEASE_CONFIGS "${_RELEASE_CONFIGS}")
      install_qt4_plugin_path("${plugin_release}" "${executable}" "${copy}" "${installed_plugin_path_var}_release" "${plugins_dir}" "${component}" "${_RELEASE_CONFIGS}")
      install_qt4_plugin_path("${plugin_debug}" "${executable}" "${copy}" "${installed_plugin_path_var}_debug" "${plugins_dir}" "${component}" "Debug")
      unset(_RELEASE_CONFIGS)

      if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
        set(${installed_plugin_path_var} ${${installed_plugin_path_var}_debug})
      else()
        set(${installed_plugin_path_var} ${${installed_plugin_path_var}_release})
      endif()
    else()
      install_qt4_plugin_path("${plugin_release}" "${executable}" "${copy}" "${installed_plugin_path_var}" "${plugins_dir}" "${component}")
    endif()
  endif()
  set(${installed_plugin_path_var} ${${installed_plugin_path_var}} PARENT_SCOPE)
endfunction()

function(install_qt4_executable executable)
  unset(qtplugins)
  if(ARGC GREATER 1)
    set(qtplugins ${ARGV1})
  endif()
  unset(libs)
  if(ARGC GREATER 2)
    set(libs ${ARGV2})
  endif()
  unset(dirs)
  if(ARGC GREATER 3)
    set(dirs ${ARGV3})
  endif()
  unset(plugins_dir)
  if(ARGC GREATER 4)
    set(plugins_dir ${ARGV4})
  endif()
  unset(request_qt_conf)
  if(ARGC GREATER 5)
    set(request_qt_conf ${ARGV5})
  endif()
  unset(component)
  if(ARGC GREATER 6)
    set(component ${ARGV6})
  endif()

  if(QT_LIBRARY_DIR)
    list(APPEND dirs "${QT_LIBRARY_DIR}")
  endif()
  if(QT_BINARY_DIR)
    list(APPEND dirs "${QT_BINARY_DIR}")
  endif()
  if(component)
    set(component COMPONENT ${component})
  else()
    unset(component)
  endif()

  get_filename_component(executable_absolute "${executable}" ABSOLUTE)
  if(EXISTS "${QT_QTCORE_LIBRARY_RELEASE}")
    gp_file_type("${executable_absolute}" "${QT_QTCORE_LIBRARY_RELEASE}" qtcore_type)
  elseif(EXISTS "${QT_QTCORE_LIBRARY_DEBUG}")
    gp_file_type("${executable_absolute}" "${QT_QTCORE_LIBRARY_DEBUG}" qtcore_type)
  endif()
  if(qtcore_type STREQUAL "system")
    set(qt_plugins_dir "")
  endif()

  if(QT_IS_STATIC)
    message(WARNING "Qt built statically: not installing plugins.")
  else()
    foreach(plugin ${qtplugins})
      set(installed_plugin_paths "")
      install_qt4_plugin("${plugin}" "${executable}" 0 installed_plugin_paths "${plugins_dir}" "${component}")
      list(APPEND libs ${installed_plugin_paths})
    endforeach()
  endif()

  resolve_qt4_paths(libs "")

  install(CODE
"include(\"${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DeployQt4.cmake\")
set(BU_CHMOD_BUNDLE_ITEMS TRUE)
fixup_qt4_executable(\"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${executable}\" \"\" \"${libs}\" \"${dirs}\" \"${plugins_dir}\" \"${request_qt_conf}\")"
          ${component}
  )
endfunction()
