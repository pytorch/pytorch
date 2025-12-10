# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGTK2
--------

.. note::

  This module is intended specifically for GTK version 2.x, which is obsolete
  and no longer maintained.  Use the latest supported GTK version and
  :module:`FindPkgConfig` module to find GTK in CMake instead of this module.
  For example:

  .. code-block:: cmake

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK REQUIRED IMPORTED_TARGET gtk4>=4.14)
    target_link_libraries(example PRIVATE PkgConfig::GTK)

Finds the GTK widget libraries and several of its other optional components:

.. code-block:: cmake

  find_package(GTK2 [<version>] [COMPONENTS <components>...] [...])

GTK is a multi-platform toolkit for creating graphical user interfaces.

Components
^^^^^^^^^^

This module supports optional components, which can be specified with the
:command:`find_package` command:

.. code-block:: cmake

  find_package(GTK2 [COMPONENTS <components>...])

Supported components include:

.. hlist::

  * ``atk``
  * ``atkmm``
  * ``cairo``
  * ``cairomm``
  * ``gdk_pixbuf``
  * ``gdk``
  * ``gdkmm``
  * ``gio``
  * ``giomm``
  * ``glade``
  * ``glademm``
  * ``glib``
  * ``glibmm``
  * ``gmodule``
  * ``gobject``
  * ``gthread``
  * ``gtk``
  * ``gtkmm``
  * ``pango``
  * ``pangocairo``
  * ``pangoft2``
  * ``pangomm``
  * ``pangoxft``
  * ``sigc``

* .. versionadded:: 3.16.7
    ``harfbuzz``

If no components are specified, module by default searches for the ``gtk``
component.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets` (subject to
component selection):

``GTK2::<component>``
  Target encapsulating the specified GTK component usage requirements,
  available if GTK and this component are found.  The ``<component>`` should
  be written in the same case, as listed above.  For example, use
  ``GTK2::gtk`` for the ``gtk`` component, or ``GTK2::gdk_pixbuf`` for the
  ``gdk_pixbuf`` component, etc.

``GTK2::sigc++``
  .. versionadded:: 3.5

  Target encapsulating the usage requirements to enable c++11 on its dependents
  when using sigc++ 2.5.1 or higher.  This target is automatically applied to
  dependent targets as needed.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``GTK2_FOUND``
  Boolean indicating whether (the requested version of) GTK 2 and all
  specified components were found.
``GTK2_VERSION``
  The version of GTK found (x.y.z).
``GTK2_MAJOR_VERSION``
  The major version of GTK found.
``GTK2_MINOR_VERSION``
  The minor version of GTK found.
``GTK2_PATCH_VERSION``
  The patch version of GTK found.
``GTK2_INCLUDE_DIRS``
  Include directories containing headers needed to use GTK.
``GTK2_LIBRARIES``
  Libraries needed to link against to use GTK.
``GTK2_TARGETS``
  .. versionadded:: 3.5

  A list of all defined imported targets.
``GTK2_DEFINITIONS``
  Additional compiler flags needed to use GTK.

Input Variables
^^^^^^^^^^^^^^^

This module accepts the following optional variables before calling the
``find_package(GTK2)``:

``GTK2_DEBUG``
  Boolean variable that enables verbose debugging output of this module.

``GTK2_ADDITIONAL_SUFFIXES``
  A list of additional path suffixes to search for include files.

``GTK2_USE_IMPORTED_TARGETS``
  .. versionadded:: 3.5

  When this variable is set to boolean true, ``GTK2_LIBRARIES`` variable will
  contain a list imported targets instead of library paths.

Examples
^^^^^^^^

Examples: Finding GTK version 2
"""""""""""""""""""""""""""""""

Call :command:`find_package` once.  Here are some examples to pick from.

Require GTK 2.6 or later:

.. code-block:: cmake

  find_package(GTK2 2.6 REQUIRED COMPONENTS gtk)

Require GTK 2.10 or later and its Glade component:

.. code-block:: cmake

  find_package(GTK2 2.10 REQUIRED COMPONENTS gtk glade)

Search for GTK/GTKMM 2.8 or later:

.. code-block:: cmake

  find_package(GTK2 2.8 COMPONENTS gtk gtkmm)

Finding GTK 2 and linking it to a project target:

.. code-block:: cmake

  find_package(GTK2)
  add_executable(mygui mygui.cc)
  target_link_libraries(mygui PRIVATE GTK2::gtk)

Examples: Finding GTK version 3 or later
""""""""""""""""""""""""""""""""""""""""

Finding GTK 3 with :module:`FindPkgConfig` instead of this module:

.. code-block:: cmake

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0>=3.14)
  target_link_libraries(example PRIVATE PkgConfig::GTK3)

Or similarly to find GTK 4:

.. code-block:: cmake

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GTK4 REQUIRED IMPORTED_TARGET gtk4>=4.14)
  target_link_libraries(example PRIVATE PkgConfig::GTK4)
#]=======================================================================]

# Version 1.6 (CMake 3.0)
#   * Create targets for each library
#   * Do not link libfreetype
# Version 1.5 (CMake 2.8.12)
#   * 14236: Detect gthread library
#            Detect pangocairo on windows
#            Detect pangocairo with gtk module instead of with gtkmm
#   * 14259: Use vc100 libraries with VS 11
#   * 14260: Export a GTK2_DEFINITIONS variable to set /vd2 when appropriate
#            (i.e. MSVC)
#   * Use the optimized/debug syntax for _LIBRARY and _LIBRARIES variables when
#     appropriate. A new set of _RELEASE variables was also added.
#   * Remove GTK2_SKIP_MARK_AS_ADVANCED option, as now the variables are
#     marked as advanced by SelectLibraryConfigurations
#   * Detect gmodule, pangoft2 and pangoxft libraries
# Version 1.4 (10/4/2012) (CMake 2.8.10)
#   * 12596: Missing paths for FindGTK2 on NetBSD
#   * 12049: Fixed detection of GTK include files in the lib folder on
#            multiarch systems.
# Version 1.3 (11/9/2010) (CMake 2.8.4)
#   * 11429: Add support for detecting GTK2 built with Visual Studio 10.
#            Thanks to Vincent Levesque for the patch.
# Version 1.2 (8/30/2010) (CMake 2.8.3)
#   * Merge patch for detecting gdk-pixbuf library (split off
#     from core GTK in 2.21).  Thanks to Vincent Untz for the patch
#     and Ricardo Cruz for the heads up.
# Version 1.1 (8/19/2010) (CMake 2.8.3)
#   * Add support for detecting GTK2 under macports (thanks to Gary Kramlich)
# Version 1.0 (8/12/2010) (CMake 2.8.3)
#   * Add support for detecting new pangommconfig.h header file
#     (Thanks to Sune Vuorela & the Debian Project for the patch)
#   * Add support for detecting fontconfig.h header
#   * Call find_package(Freetype) since it's required
#   * Add support for allowing users to add additional library directories
#     via the GTK2_ADDITIONAL_SUFFIXES variable (kind of a future-kludge in
#     case the GTK developers change versions on any of the directories in the
#     future).
# Version 0.8 (1/4/2010)
#   * Get module working under MacOSX fink by adding /sw/include, /sw/lib
#     to PATHS and the gobject library
# Version 0.7 (3/22/09)
#   * Checked into CMake CVS
#   * Added versioning support
#   * Module now defaults to searching for GTK if COMPONENTS not specified.
#   * Added HKCU prior to HKLM registry key and GTKMM specific environment
#      variable as per mailing list discussion.
#   * Added lib64 to include search path and a few other search paths where GTK
#      may be installed on Unix systems.
#   * Switched to lowercase CMake commands
#   * Prefaced internal variables with _GTK2 to prevent collision
#   * Changed internal macros to functions
#   * Enhanced documentation
# Version 0.6 (1/8/08)
#   Added GTK2_SKIP_MARK_AS_ADVANCED option
# Version 0.5 (12/19/08)
#   Second release to cmake mailing list

#=============================================================
# _GTK2_GET_VERSION
# Internal function to parse the version number in gtkversion.h
#   _OUT_major = Major version number
#   _OUT_minor = Minor version number
#   _OUT_micro = Micro version number
#   _gtkversion_hdr = Header file to parse
#=============================================================

include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)

function(_GTK2_GET_VERSION _OUT_major _OUT_minor _OUT_micro _gtkversion_hdr)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
    file(STRINGS ${_gtkversion_hdr} _contents REGEX "#define GTK_M[A-Z]+_VERSION[ \t]+")
    cmake_policy(POP)
    if(_contents)
        string(REGEX REPLACE ".*#define GTK_MAJOR_VERSION[ \t]+\\(([0-9]+)\\).*" "\\1" ${_OUT_major} "${_contents}")
        string(REGEX REPLACE ".*#define GTK_MINOR_VERSION[ \t]+\\(([0-9]+)\\).*" "\\1" ${_OUT_minor} "${_contents}")
        string(REGEX REPLACE ".*#define GTK_MICRO_VERSION[ \t]+\\(([0-9]+)\\).*" "\\1" ${_OUT_micro} "${_contents}")

        if(NOT ${_OUT_major} MATCHES "[0-9]+")
            message(FATAL_ERROR "Version parsing failed for GTK2_MAJOR_VERSION!")
        endif()
        if(NOT ${_OUT_minor} MATCHES "[0-9]+")
            message(FATAL_ERROR "Version parsing failed for GTK2_MINOR_VERSION!")
        endif()
        if(NOT ${_OUT_micro} MATCHES "[0-9]+")
            message(FATAL_ERROR "Version parsing failed for GTK2_MICRO_VERSION!")
        endif()

        set(${_OUT_major} ${${_OUT_major}} PARENT_SCOPE)
        set(${_OUT_minor} ${${_OUT_minor}} PARENT_SCOPE)
        set(${_OUT_micro} ${${_OUT_micro}} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Include file ${_gtkversion_hdr} does not exist")
    endif()
endfunction()


#=============================================================
# _GTK2_SIGCXX_GET_VERSION
# Internal function to parse the version number in
# sigc++config.h
#   _OUT_major = Major version number
#   _OUT_minor = Minor version number
#   _OUT_micro = Micro version number
#   _sigcxxversion_hdr = Header file to parse
#=============================================================

function(_GTK2_SIGCXX_GET_VERSION _OUT_major _OUT_minor _OUT_micro _sigcxxversion_hdr)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
    file(STRINGS ${_sigcxxversion_hdr} _contents REGEX "#define SIGCXX_M[A-Z]+_VERSION[ \t]+")
    cmake_policy(POP)
    if(_contents)
        string(REGEX REPLACE ".*#define SIGCXX_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1" ${_OUT_major} "${_contents}")
        string(REGEX REPLACE ".*#define SIGCXX_MINOR_VERSION[ \t]+([0-9]+).*" "\\1" ${_OUT_minor} "${_contents}")
        string(REGEX REPLACE ".*#define SIGCXX_MICRO_VERSION[ \t]+([0-9]+).*" "\\1" ${_OUT_micro} "${_contents}")

        if(NOT ${_OUT_major} MATCHES "[0-9]+")
            message(FATAL_ERROR "Version parsing failed for SIGCXX_MAJOR_VERSION!")
        endif()
        if(NOT ${_OUT_minor} MATCHES "[0-9]+")
            message(FATAL_ERROR "Version parsing failed for SIGCXX_MINOR_VERSION!")
        endif()
        if(NOT ${_OUT_micro} MATCHES "[0-9]+")
            message(FATAL_ERROR "Version parsing failed for SIGCXX_MICRO_VERSION!")
        endif()

        set(${_OUT_major} ${${_OUT_major}} PARENT_SCOPE)
        set(${_OUT_minor} ${${_OUT_minor}} PARENT_SCOPE)
        set(${_OUT_micro} ${${_OUT_micro}} PARENT_SCOPE)
    else()
        # The header does not have the version macros; assume it is ``0.0.0``.
        set(${_OUT_major} 0)
        set(${_OUT_minor} 0)
        set(${_OUT_micro} 0)
    endif()
endfunction()


#=============================================================
# _GTK2_FIND_INCLUDE_DIR
# Internal function to find the GTK include directories
#   _var = variable to set (_INCLUDE_DIR is appended)
#   _hdr = header file to look for
#=============================================================
function(_GTK2_FIND_INCLUDE_DIR _var _hdr)

    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}] "
                       "_GTK2_FIND_INCLUDE_DIR( ${_var} ${_hdr} )")
    endif()

    set(_gtk_packages
        # If these ever change, things will break.
        ${GTK2_ADDITIONAL_SUFFIXES}
        glibmm-2.4
        glib-2.0
        atk-1.0
        atkmm-1.6
        cairo
        cairomm-1.0
        gdk-pixbuf-2.0
        gdkmm-2.4
        giomm-2.4
        gtk-2.0
        gtkmm-2.4
        libglade-2.0
        libglademm-2.4
        harfbuzz
        pango-1.0
        pangomm-1.4
        sigc++-2.0
    )

    #
    # NOTE: The following suffixes cause searching for header files in both of
    # these directories:
    #         /usr/include/<pkg>
    #         /usr/lib/<pkg>/include
    #

    set(_suffixes)
    foreach(_d ${_gtk_packages})
        list(APPEND _suffixes ${_d})
        list(APPEND _suffixes ${_d}/include) # for /usr/lib/gtk-2.0/include
    endforeach()

    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}]     "
                       "include suffixes = ${_suffixes}")
    endif()

    if(CMAKE_LIBRARY_ARCHITECTURE)
      set(_gtk2_arch_dir /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE})
      if(GTK2_DEBUG)
        message(STATUS "Adding ${_gtk2_arch_dir} to search path for multiarch support")
      endif()
    endif()
    find_path(GTK2_${_var}_INCLUDE_DIR ${_hdr}
        PATHS
            ${PC_GLIB2_INCLUDEDIR}
            ${PC_GLIB2_LIBDIR}
            ${PC_GTK2_INCLUDEDIR}
            ${PC_GTK2_LIBDIR}
            ${_gtk2_arch_dir}
            /usr/local/libx32
            /usr/local/lib64
            /usr/local/lib
            /usr/libx32
            /usr/lib64
            /usr/lib
            /opt/gnome/include
            /opt/gnome/lib
            /opt/openwin/include
            /usr/openwin/lib
            /sw/lib
            /opt/local/lib
            /opt/homebrew/lib
            /usr/pkg/lib
            /usr/pkg/include/glib
            $ENV{GTKMM_BASEPATH}/include
            $ENV{GTKMM_BASEPATH}/lib
            [HKEY_CURRENT_USER\\SOFTWARE\\gtkmm\\2.4;Path]/include
            [HKEY_CURRENT_USER\\SOFTWARE\\gtkmm\\2.4;Path]/lib
            [HKEY_LOCAL_MACHINE\\SOFTWARE\\gtkmm\\2.4;Path]/include
            [HKEY_LOCAL_MACHINE\\SOFTWARE\\gtkmm\\2.4;Path]/lib
        PATH_SUFFIXES
            ${_suffixes}
    )
    mark_as_advanced(GTK2_${_var}_INCLUDE_DIR)

    if(GTK2_${_var}_INCLUDE_DIR)
        set(GTK2_INCLUDE_DIRS ${GTK2_INCLUDE_DIRS} ${GTK2_${_var}_INCLUDE_DIR} PARENT_SCOPE)
    endif()

endfunction()

#=============================================================
# _GTK2_FIND_LIBRARY
# Internal function to find libraries packaged with GTK2
#   _var = library variable to create (_LIBRARY is appended)
#=============================================================
function(_GTK2_FIND_LIBRARY _var _lib _expand_vc _append_version)

    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}] "
                       "_GTK2_FIND_LIBRARY( ${_var} ${_lib} ${_expand_vc} ${_append_version} )")
    endif()

    # Not GTK versions per se but the versions encoded into Windows
    # import libraries (GtkMM 2.14.1 has a gtkmm-vc80-2_4.lib for example)
    # Also the MSVC libraries use _ for . (this is handled below)
    set(_versions 2.20 2.18 2.16 2.14 2.12
                  2.10  2.8  2.6  2.4  2.2 2.0
                  1.20 1.18 1.16 1.14 1.12
                  1.10  1.8  1.6  1.4  1.2 1.0)

    set(_library)
    set(_library_d)

    set(_library ${_lib})

    if(_expand_vc AND MSVC)
        # Add vc80/vc90/vc100 midfixes
        if(MSVC_TOOLSET_VERSION LESS 110)
            set(_library   ${_library}-vc${MSVC_TOOLSET_VERSION})
        else()
            # Up to gtkmm-win 2.22.0-2 there are no vc110 libraries but vc100 can be used
            set(_library ${_library}-vc100)
        endif()
        set(_library_d ${_library}-d)
    endif()

    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}]     "
                       "After midfix addition = ${_library} and ${_library_d}")
    endif()

    set(_lib_list)
    set(_libd_list)
    if(_append_version)
        foreach(_ver ${_versions})
            list(APPEND _lib_list  "${_library}-${_ver}")
            list(APPEND _libd_list "${_library_d}-${_ver}")
        endforeach()
    else()
        set(_lib_list ${_library})
        set(_libd_list ${_library_d})
    endif()

    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}]     "
                       "library list = ${_lib_list} and library debug list = ${_libd_list}")
    endif()

    # For some silly reason the MSVC libraries use _ instead of .
    # in the version fields
    if(_expand_vc AND MSVC)
        set(_no_dots_lib_list)
        set(_no_dots_libd_list)
        foreach(_l ${_lib_list})
            string(REPLACE "." "_" _no_dots_library ${_l})
            list(APPEND _no_dots_lib_list ${_no_dots_library})
        endforeach()
        # And for debug
        set(_no_dots_libsd_list)
        foreach(_l ${_libd_list})
            string(REPLACE "." "_" _no_dots_libraryd ${_l})
            list(APPEND _no_dots_libd_list ${_no_dots_libraryd})
        endforeach()

        # Copy list back to original names
        set(_lib_list ${_no_dots_lib_list})
        set(_libd_list ${_no_dots_libd_list})
    endif()

    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}]     "
                       "While searching for GTK2_${_var}_LIBRARY, our proposed library list is ${_lib_list}")
    endif()

    find_library(GTK2_${_var}_LIBRARY_RELEASE
        NAMES ${_lib_list}
        PATHS
            /opt/gnome/lib
            /usr/openwin/lib
            $ENV{GTKMM_BASEPATH}/lib
            [HKEY_CURRENT_USER\\SOFTWARE\\gtkmm\\2.4;Path]/lib
            [HKEY_LOCAL_MACHINE\\SOFTWARE\\gtkmm\\2.4;Path]/lib
        )

    if(_expand_vc AND MSVC)
        if(GTK2_DEBUG)
            message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}]     "
                           "While searching for GTK2_${_var}_LIBRARY_DEBUG our proposed library list is ${_libd_list}")
        endif()

        find_library(GTK2_${_var}_LIBRARY_DEBUG
            NAMES ${_libd_list}
            PATHS
            $ENV{GTKMM_BASEPATH}/lib
            [HKEY_CURRENT_USER\\SOFTWARE\\gtkmm\\2.4;Path]/lib
            [HKEY_LOCAL_MACHINE\\SOFTWARE\\gtkmm\\2.4;Path]/lib
        )
    endif()

    select_library_configurations(GTK2_${_var})

    set(GTK2_${_var}_LIBRARY ${GTK2_${_var}_LIBRARY} PARENT_SCOPE)
    set(GTK2_${_var}_FOUND ${GTK2_${_var}_FOUND} PARENT_SCOPE)

    if(GTK2_${_var}_FOUND)
        set(GTK2_LIBRARIES ${GTK2_LIBRARIES} ${GTK2_${_var}_LIBRARY})
        set(GTK2_LIBRARIES ${GTK2_LIBRARIES} PARENT_SCOPE)
    endif()

    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}]     "
                       "GTK2_${_var}_LIBRARY_RELEASE = \"${GTK2_${_var}_LIBRARY_RELEASE}\"")
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}]     "
                       "GTK2_${_var}_LIBRARY_DEBUG   = \"${GTK2_${_var}_LIBRARY_DEBUG}\"")
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}]     "
                       "GTK2_${_var}_LIBRARY         = \"${GTK2_${_var}_LIBRARY}\"")
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}]     "
                       "GTK2_${_var}_FOUND           = \"${GTK2_${_var}_FOUND}\"")
    endif()

endfunction()


function(_GTK2_ADD_TARGET_DEPENDS_INTERNAL _var _property)
    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}] "
                       "_GTK2_ADD_TARGET_DEPENDS_INTERNAL( ${_var} ${_property} )")
    endif()

    string(TOLOWER "${_var}" _basename)

    if (TARGET GTK2::${_basename})
        foreach(_depend ${ARGN})
            set(_valid_depends)
            if (TARGET GTK2::${_depend})
                list(APPEND _valid_depends GTK2::${_depend})
            endif()
            if (_valid_depends)
                set_property(TARGET GTK2::${_basename} APPEND PROPERTY ${_property} "${_valid_depends}")
            endif()
            set(_valid_depends)
        endforeach()
    endif()
endfunction()

function(_GTK2_ADD_TARGET_DEPENDS _var)
    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}] "
                       "_GTK2_ADD_TARGET_DEPENDS( ${_var} )")
    endif()

    string(TOLOWER "${_var}" _basename)

    if(TARGET GTK2::${_basename})
        get_target_property(_configs GTK2::${_basename} IMPORTED_CONFIGURATIONS)
        _GTK2_ADD_TARGET_DEPENDS_INTERNAL(${_var} INTERFACE_LINK_LIBRARIES ${ARGN})
        foreach(_config ${_configs})
            _GTK2_ADD_TARGET_DEPENDS_INTERNAL(${_var} IMPORTED_LINK_INTERFACE_LIBRARIES_${_config} ${ARGN})
        endforeach()
    endif()
endfunction()

function(_GTK2_ADD_TARGET_INCLUDE_DIRS _var)
    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}] "
                       "_GTK2_ADD_TARGET_INCLUDE_DIRS( ${_var} )")
    endif()

    string(TOLOWER "${_var}" _basename)

    if(TARGET GTK2::${_basename})
        foreach(_include ${ARGN})
            set_property(TARGET GTK2::${_basename} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${_include}")
        endforeach()
    endif()
endfunction()

#=============================================================
# _GTK2_ADD_TARGET
# Internal function to create targets for GTK2
#   _var = target to create
#=============================================================
function(_GTK2_ADD_TARGET _var)
    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}] "
                       "_GTK2_ADD_TARGET( ${_var} )")
    endif()

    string(TOLOWER "${_var}" _basename)

    cmake_parse_arguments(_${_var} "" "" "GTK2_DEPENDS;GTK2_OPTIONAL_DEPENDS;OPTIONAL_INCLUDES" ${ARGN})

    if(GTK2_${_var}_FOUND)
        if(NOT TARGET GTK2::${_basename})
            # Do not create the target if dependencies are missing
            foreach(_dep ${_${_var}_GTK2_DEPENDS})
                if(NOT TARGET GTK2::${_dep})
                    return()
                endif()
            endforeach()

            add_library(GTK2::${_basename} UNKNOWN IMPORTED)

            if(GTK2_${_var}_LIBRARY_RELEASE)
                set_property(TARGET GTK2::${_basename} APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
                set_property(TARGET GTK2::${_basename}        PROPERTY IMPORTED_LOCATION_RELEASE "${GTK2_${_var}_LIBRARY_RELEASE}" )
            endif()

            if(GTK2_${_var}_LIBRARY_DEBUG)
                set_property(TARGET GTK2::${_basename} APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
                set_property(TARGET GTK2::${_basename}        PROPERTY IMPORTED_LOCATION_DEBUG "${GTK2_${_var}_LIBRARY_DEBUG}" )
            endif()

            if(GTK2_${_var}_INCLUDE_DIR)
                set_property(TARGET GTK2::${_basename} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${GTK2_${_var}_INCLUDE_DIR}")
            endif()

            if(GTK2_${_var}CONFIG_INCLUDE_DIR AND NOT "x${GTK2_${_var}CONFIG_INCLUDE_DIR}" STREQUAL "x${GTK2_${_var}_INCLUDE_DIR}")
                set_property(TARGET GTK2::${_basename} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${GTK2_${_var}CONFIG_INCLUDE_DIR}")
            endif()

            if(GTK2_DEFINITIONS)
                set_property(TARGET GTK2::${_basename} PROPERTY INTERFACE_COMPILE_DEFINITIONS "${GTK2_DEFINITIONS}")
            endif()

            if(_${_var}_GTK2_DEPENDS)
                _GTK2_ADD_TARGET_DEPENDS(${_var} ${_${_var}_GTK2_DEPENDS} ${_${_var}_GTK2_OPTIONAL_DEPENDS})
            endif()

            if(_${_var}_OPTIONAL_INCLUDES)
                foreach(_D ${_${_var}_OPTIONAL_INCLUDES})
                    if(_D)
                        _GTK2_ADD_TARGET_INCLUDE_DIRS(${_var} ${_D})
                    endif()
                endforeach()
            endif()
        endif()

        set(GTK2_TARGETS ${GTK2_TARGETS} GTK2::${_basename})
        set(GTK2_TARGETS ${GTK2_TARGETS} PARENT_SCOPE)

        if(GTK2_USE_IMPORTED_TARGETS)
            set(GTK2_${_var}_LIBRARY GTK2::${_basename} PARENT_SCOPE)
        endif()

    endif()
endfunction()



#=============================================================

#
# main()
#

set(GTK2_FOUND)
set(GTK2_INCLUDE_DIRS)
set(GTK2_LIBRARIES)
set(GTK2_TARGETS)
set(GTK2_DEFINITIONS)

if(NOT GTK2_FIND_COMPONENTS)
    # Assume they only want GTK
    set(GTK2_FIND_COMPONENTS gtk)
endif()

# Retrieve LIBDIR from the GTK2 and GLIB2 pkg-config files, which are
# used to compute the arch-specific include prefixes. While at it,
# also retrieve their INCLUDEDIR, to accommodate non-standard layouts.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_GTK2 QUIET gtk+-2.0)
  if(PC_GTK2_FOUND)
    pkg_get_variable(PC_GTK2_INCLUDEDIR gtk+-2.0 includedir)
    pkg_get_variable(PC_GTK2_LIBDIR gtk+-2.0 libdir)
  endif()
  pkg_check_modules(PC_GLIB2 QUIET glib-2.0)
  if(PC_GLIB2_FOUND)
    pkg_get_variable(PC_GLIB2_INCLUDEDIR glib-2.0 includedir)
    pkg_get_variable(PC_GLIB2_LIBDIR glib-2.0 libdir)
  endif()
endif()

#
# If specified, enforce version number
#
if(GTK2_FIND_VERSION)
    set(GTK2_FAILED_VERSION_CHECK true)
    if(GTK2_DEBUG)
        message(STATUS "[FindGTK2.cmake:${CMAKE_CURRENT_LIST_LINE}] "
                       "Searching for version ${GTK2_FIND_VERSION}")
    endif()
    _GTK2_FIND_INCLUDE_DIR(GTK gtk/gtk.h)
    if(GTK2_GTK_INCLUDE_DIR)
        _GTK2_GET_VERSION(GTK2_MAJOR_VERSION
                          GTK2_MINOR_VERSION
                          GTK2_PATCH_VERSION
                          ${GTK2_GTK_INCLUDE_DIR}/gtk/gtkversion.h)
        set(GTK2_VERSION
            ${GTK2_MAJOR_VERSION}.${GTK2_MINOR_VERSION}.${GTK2_PATCH_VERSION})
        if(GTK2_FIND_VERSION_EXACT)
            if(GTK2_VERSION VERSION_EQUAL GTK2_FIND_VERSION)
                set(GTK2_FAILED_VERSION_CHECK false)
            endif()
        else()
            if(GTK2_VERSION VERSION_EQUAL   GTK2_FIND_VERSION OR
               GTK2_VERSION VERSION_GREATER GTK2_FIND_VERSION)
                set(GTK2_FAILED_VERSION_CHECK false)
            endif()
        endif()
    else()
        # If we can't find the GTK include dir, we can't do version checking
        if(GTK2_FIND_REQUIRED AND NOT GTK2_FIND_QUIETLY)
            message(FATAL_ERROR "Could not find GTK2 include directory")
        endif()
        return()
    endif()

    if(GTK2_FAILED_VERSION_CHECK)
        if(GTK2_FIND_REQUIRED AND NOT GTK2_FIND_QUIETLY)
            if(GTK2_FIND_VERSION_EXACT)
                message(FATAL_ERROR "GTK2 version check failed.  Version ${GTK2_VERSION} was found, version ${GTK2_FIND_VERSION} is needed exactly.")
            else()
                message(FATAL_ERROR "GTK2 version check failed.  Version ${GTK2_VERSION} was found, at least version ${GTK2_FIND_VERSION} is required")
            endif()
        endif()

        # If the version check fails, exit out of the module here
        return()
    endif()
endif()

#
# On MSVC, according to https://wiki.gnome.org/gtkmm/MSWindows, the /vd2 flag needs to be
# passed to the compiler in order to use gtkmm
#
if(MSVC)
    foreach(_GTK2_component ${GTK2_FIND_COMPONENTS})
        if(_GTK2_component STREQUAL "gtkmm")
            set(GTK2_DEFINITIONS "/vd2")
        elseif(_GTK2_component STREQUAL "glademm")
            set(GTK2_DEFINITIONS "/vd2")
        endif()
    endforeach()
endif()

#
# Find all components
#

find_package(Freetype QUIET)
if(FREETYPE_INCLUDE_DIR_ft2build AND FREETYPE_INCLUDE_DIR_freetype2)
    list(APPEND GTK2_INCLUDE_DIRS ${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2})
endif()

foreach(_GTK2_component ${GTK2_FIND_COMPONENTS})
    if(_GTK2_component STREQUAL "gtk")
        # Left for compatibility with previous versions.
        _GTK2_FIND_INCLUDE_DIR(FONTCONFIG fontconfig/fontconfig.h)
        _GTK2_FIND_INCLUDE_DIR(X11 X11/Xlib.h)

        _GTK2_FIND_INCLUDE_DIR(GLIB glib.h)
        _GTK2_FIND_INCLUDE_DIR(GLIBCONFIG glibconfig.h)
        _GTK2_FIND_LIBRARY    (GLIB glib false true)
        _GTK2_ADD_TARGET      (GLIB)

        _GTK2_FIND_INCLUDE_DIR(GOBJECT glib-object.h)
        _GTK2_FIND_LIBRARY    (GOBJECT gobject false true)
        _GTK2_ADD_TARGET      (GOBJECT GTK2_DEPENDS glib)

        _GTK2_FIND_INCLUDE_DIR(ATK atk/atk.h)
        _GTK2_FIND_LIBRARY    (ATK atk false true)
        _GTK2_ADD_TARGET      (ATK GTK2_DEPENDS gobject glib)

        _GTK2_FIND_LIBRARY    (GIO gio false true)
        _GTK2_ADD_TARGET      (GIO GTK2_DEPENDS gobject glib)

        _GTK2_FIND_LIBRARY    (GTHREAD gthread false true)
        _GTK2_ADD_TARGET      (GTHREAD GTK2_DEPENDS glib)

        _GTK2_FIND_LIBRARY    (GMODULE gmodule false true)
        _GTK2_ADD_TARGET      (GMODULE GTK2_DEPENDS glib)

        _GTK2_FIND_INCLUDE_DIR(GDK_PIXBUF gdk-pixbuf/gdk-pixbuf.h)
        _GTK2_FIND_LIBRARY    (GDK_PIXBUF gdk_pixbuf false true)
        _GTK2_ADD_TARGET      (GDK_PIXBUF GTK2_DEPENDS gobject glib)

        _GTK2_FIND_INCLUDE_DIR(CAIRO cairo.h)
        _GTK2_FIND_LIBRARY    (CAIRO cairo false false)
        _GTK2_ADD_TARGET      (CAIRO)

        _GTK2_FIND_INCLUDE_DIR(HARFBUZZ hb.h)
        _GTK2_FIND_LIBRARY    (HARFBUZZ harfbuzz false false)
        _GTK2_ADD_TARGET      (HARFBUZZ)

        _GTK2_FIND_INCLUDE_DIR(PANGO pango/pango.h)
        _GTK2_FIND_LIBRARY    (PANGO pango false true)
        _GTK2_ADD_TARGET      (PANGO GTK2_DEPENDS gobject glib
                                     GTK2_OPTIONAL_DEPENDS harfbuzz)

        _GTK2_FIND_LIBRARY    (PANGOCAIRO pangocairo false true)
        _GTK2_ADD_TARGET      (PANGOCAIRO GTK2_DEPENDS pango cairo gobject glib)

        _GTK2_FIND_LIBRARY    (PANGOFT2 pangoft2 false true)
        _GTK2_ADD_TARGET      (PANGOFT2 GTK2_DEPENDS pango gobject glib
                                        OPTIONAL_INCLUDES ${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2}
                                                          ${GTK2_FONTCONFIG_INCLUDE_DIR}
                                                          ${GTK2_X11_INCLUDE_DIR})

        _GTK2_FIND_LIBRARY    (PANGOXFT pangoxft false true)
        _GTK2_ADD_TARGET      (PANGOXFT GTK2_DEPENDS pangoft2 pango gobject glib
                                        OPTIONAL_INCLUDES ${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2}
                                                          ${GTK2_FONTCONFIG_INCLUDE_DIR}
                                                          ${GTK2_X11_INCLUDE_DIR})

        _GTK2_FIND_INCLUDE_DIR(GDK gdk/gdk.h)
        _GTK2_FIND_INCLUDE_DIR(GDKCONFIG gdkconfig.h)
        if(UNIX)
            if(APPLE)
                _GTK2_FIND_LIBRARY    (GDK gdk-quartz false true)
            endif()
            _GTK2_FIND_LIBRARY    (GDK gdk-x11 false true)
        else()
            _GTK2_FIND_LIBRARY    (GDK gdk-win32 false true)
        endif()
        _GTK2_ADD_TARGET (GDK GTK2_DEPENDS pango gdk_pixbuf gobject glib
                              GTK2_OPTIONAL_DEPENDS pangocairo cairo)

        _GTK2_FIND_INCLUDE_DIR(GTK gtk/gtk.h)
        if(UNIX)
            if(APPLE)
                _GTK2_FIND_LIBRARY    (GTK gtk-quartz false true)
            endif()
            _GTK2_FIND_LIBRARY    (GTK gtk-x11 false true)
        else()
            _GTK2_FIND_LIBRARY    (GTK gtk-win32 false true)
        endif()
        _GTK2_ADD_TARGET (GTK GTK2_DEPENDS gdk atk pangoft2 pango gdk_pixbuf gthread gobject glib
                              GTK2_OPTIONAL_DEPENDS gio pangocairo cairo)

    elseif(_GTK2_component STREQUAL "gtkmm")

        _GTK2_FIND_INCLUDE_DIR(SIGC++ sigc++/sigc++.h)
        _GTK2_FIND_INCLUDE_DIR(SIGC++CONFIG sigc++config.h)
        _GTK2_FIND_LIBRARY    (SIGC++ sigc true true)
        _GTK2_ADD_TARGET      (SIGC++)
        # Since sigc++ 2.5.1 c++11 support is required
        if(GTK2_SIGC++CONFIG_INCLUDE_DIR)
            _GTK2_SIGCXX_GET_VERSION(GTK2_SIGC++_VERSION_MAJOR
                                     GTK2_SIGC++_VERSION_MINOR
                                     GTK2_SIGC++_VERSION_MICRO
                                     ${GTK2_SIGC++CONFIG_INCLUDE_DIR}/sigc++config.h)
            if(NOT ${GTK2_SIGC++_VERSION_MAJOR}.${GTK2_SIGC++_VERSION_MINOR}.${GTK2_SIGC++_VERSION_MICRO} VERSION_LESS 2.5.1)
                # These are the features needed by clients in order to include the
                # project headers:
                set_property(TARGET GTK2::sigc++
                             PROPERTY INTERFACE_COMPILE_FEATURES cxx_alias_templates
                                                                 cxx_auto_type
                                                                 cxx_decltype
                                                                 cxx_deleted_functions
                                                                 cxx_noexcept
                                                                 cxx_nullptr
                                                                 cxx_right_angle_brackets
                                                                 cxx_rvalue_references
                                                                 cxx_variadic_templates)
            endif()
        endif()

        _GTK2_FIND_INCLUDE_DIR(GLIBMM glibmm.h)
        _GTK2_FIND_INCLUDE_DIR(GLIBMMCONFIG glibmmconfig.h)
        _GTK2_FIND_LIBRARY    (GLIBMM glibmm true true)
        _GTK2_ADD_TARGET      (GLIBMM GTK2_DEPENDS gobject sigc++ glib)

        _GTK2_FIND_INCLUDE_DIR(GIOMM giomm.h)
        _GTK2_FIND_INCLUDE_DIR(GIOMMCONFIG giommconfig.h)
        _GTK2_FIND_LIBRARY    (GIOMM giomm true true)
        _GTK2_ADD_TARGET      (GIOMM GTK2_DEPENDS gio glibmm gobject sigc++ glib)

        _GTK2_FIND_INCLUDE_DIR(ATKMM atkmm.h)
        _GTK2_FIND_INCLUDE_DIR(ATKMMCONFIG atkmmconfig.h)
        _GTK2_FIND_LIBRARY    (ATKMM atkmm true true)
        _GTK2_ADD_TARGET      (ATKMM GTK2_DEPENDS atk glibmm gobject sigc++ glib)

        _GTK2_FIND_INCLUDE_DIR(CAIROMM cairomm/cairomm.h)
        _GTK2_FIND_INCLUDE_DIR(CAIROMMCONFIG cairommconfig.h)
        _GTK2_FIND_LIBRARY    (CAIROMM cairomm true true)
        _GTK2_ADD_TARGET      (CAIROMM GTK2_DEPENDS cairo sigc++
                                       OPTIONAL_INCLUDES ${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2}
                                                         ${GTK2_FONTCONFIG_INCLUDE_DIR}
                                                         ${GTK2_X11_INCLUDE_DIR})

        _GTK2_FIND_INCLUDE_DIR(PANGOMM pangomm.h)
        _GTK2_FIND_INCLUDE_DIR(PANGOMMCONFIG pangommconfig.h)
        _GTK2_FIND_LIBRARY    (PANGOMM pangomm true true)
        _GTK2_ADD_TARGET      (PANGOMM GTK2_DEPENDS glibmm sigc++ pango gobject glib
                                       GTK2_OPTIONAL_DEPENDS cairomm pangocairo cairo
                                       OPTIONAL_INCLUDES ${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2}
                                                         ${GTK2_FONTCONFIG_INCLUDE_DIR}
                                                         ${GTK2_X11_INCLUDE_DIR})

        _GTK2_FIND_INCLUDE_DIR(GDKMM gdkmm.h)
        _GTK2_FIND_INCLUDE_DIR(GDKMMCONFIG gdkmmconfig.h)
        _GTK2_FIND_LIBRARY    (GDKMM gdkmm true true)
        _GTK2_ADD_TARGET      (GDKMM GTK2_DEPENDS pangomm gtk glibmm sigc++ gdk atk pangoft2 gdk_pixbuf pango gobject glib
                                     GTK2_OPTIONAL_DEPENDS giomm cairomm gio pangocairo cairo
                                     OPTIONAL_INCLUDES ${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2}
                                                       ${GTK2_FONTCONFIG_INCLUDE_DIR}
                                                       ${GTK2_X11_INCLUDE_DIR})

        _GTK2_FIND_INCLUDE_DIR(GTKMM gtkmm.h)
        _GTK2_FIND_INCLUDE_DIR(GTKMMCONFIG gtkmmconfig.h)
        _GTK2_FIND_LIBRARY    (GTKMM gtkmm true true)
        _GTK2_ADD_TARGET      (GTKMM GTK2_DEPENDS atkmm gdkmm pangomm gtk glibmm sigc++ gdk atk pangoft2 gdk_pixbuf pango gthread gobject glib
                                     GTK2_OPTIONAL_DEPENDS giomm cairomm gio pangocairo cairo
                                     OPTIONAL_INCLUDES ${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2}
                                                       ${GTK2_FONTCONFIG_INCLUDE_DIR}
                                                       ${GTK2_X11_INCLUDE_DIR})

    elseif(_GTK2_component STREQUAL "glade")

        _GTK2_FIND_INCLUDE_DIR(GLADE glade/glade.h)
        _GTK2_FIND_LIBRARY    (GLADE glade false true)
        _GTK2_ADD_TARGET      (GLADE GTK2_DEPENDS gtk gdk atk gio pangoft2 gdk_pixbuf pango gobject glib
                                     GTK2_OPTIONAL_DEPENDS pangocairo cairo
                                     OPTIONAL_INCLUDES ${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2}
                                                       ${GTK2_FONTCONFIG_INCLUDE_DIR}
                                                       ${GTK2_X11_INCLUDE_DIR})

    elseif(_GTK2_component STREQUAL "glademm")

        _GTK2_FIND_INCLUDE_DIR(GLADEMM libglademm.h)
        _GTK2_FIND_INCLUDE_DIR(GLADEMMCONFIG libglademmconfig.h)
        _GTK2_FIND_LIBRARY    (GLADEMM glademm true true)
        _GTK2_ADD_TARGET      (GLADEMM GTK2_DEPENDS gtkmm glade atkmm gdkmm giomm pangomm glibmm sigc++ gtk gdk atk pangoft2 gdk_pixbuf pango gthread gobject glib
                                       GTK2_OPTIONAL_DEPENDS giomm cairomm gio pangocairo cairo
                                       OPTIONAL_INCLUDES ${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2}
                                                         ${GTK2_FONTCONFIG_INCLUDE_DIR}
                                                         ${GTK2_X11_INCLUDE_DIR})

    else()
        message(FATAL_ERROR "Unknown GTK2 component ${_component}")
    endif()
endforeach()

#
# Solve for the GTK2 version if we haven't already
#
if(NOT GTK2_FIND_VERSION AND GTK2_GTK_INCLUDE_DIR)
    _GTK2_GET_VERSION(GTK2_MAJOR_VERSION
                      GTK2_MINOR_VERSION
                      GTK2_PATCH_VERSION
                      ${GTK2_GTK_INCLUDE_DIR}/gtk/gtkversion.h)
    set(GTK2_VERSION ${GTK2_MAJOR_VERSION}.${GTK2_MINOR_VERSION}.${GTK2_PATCH_VERSION})
endif()

#
# Try to enforce components
#

set(_GTK2_did_we_find_everything true)  # This gets set to GTK2_FOUND

include(FindPackageHandleStandardArgs)

foreach(_GTK2_component ${GTK2_FIND_COMPONENTS})
    string(TOUPPER ${_GTK2_component} _COMPONENT_UPPER)

    set(GTK2_${_COMPONENT_UPPER}_FIND_QUIETLY ${GTK2_FIND_QUIETLY})

    set(FPHSA_NAME_MISMATCHED 1)
    if(_GTK2_component STREQUAL "gtk")
        find_package_handle_standard_args(GTK2_${_COMPONENT_UPPER} "Some or all of the gtk libraries were not found."
            GTK2_GTK_LIBRARY
            GTK2_GTK_INCLUDE_DIR

            GTK2_GDK_INCLUDE_DIR
            GTK2_GDKCONFIG_INCLUDE_DIR
            GTK2_GDK_LIBRARY

            GTK2_GLIB_INCLUDE_DIR
            GTK2_GLIBCONFIG_INCLUDE_DIR
            GTK2_GLIB_LIBRARY
        )
    elseif(_GTK2_component STREQUAL "gtkmm")
        find_package_handle_standard_args(GTK2_${_COMPONENT_UPPER} "Some or all of the gtkmm libraries were not found."
            GTK2_GTKMM_LIBRARY
            GTK2_GTKMM_INCLUDE_DIR
            GTK2_GTKMMCONFIG_INCLUDE_DIR

            GTK2_GDKMM_INCLUDE_DIR
            GTK2_GDKMMCONFIG_INCLUDE_DIR
            GTK2_GDKMM_LIBRARY

            GTK2_GLIBMM_INCLUDE_DIR
            GTK2_GLIBMMCONFIG_INCLUDE_DIR
            GTK2_GLIBMM_LIBRARY

            FREETYPE_INCLUDE_DIR_ft2build
            FREETYPE_INCLUDE_DIR_freetype2
        )
    elseif(_GTK2_component STREQUAL "glade")
        find_package_handle_standard_args(GTK2_${_COMPONENT_UPPER} "The glade library was not found."
            GTK2_GLADE_LIBRARY
            GTK2_GLADE_INCLUDE_DIR
        )
    elseif(_GTK2_component STREQUAL "glademm")
        find_package_handle_standard_args(GTK2_${_COMPONENT_UPPER} "The glademm library was not found."
            GTK2_GLADEMM_LIBRARY
            GTK2_GLADEMM_INCLUDE_DIR
            GTK2_GLADEMMCONFIG_INCLUDE_DIR
        )
    endif()
    unset(FPHSA_NAME_MISMATCHED)

    if(NOT GTK2_${_COMPONENT_UPPER}_FOUND)
        set(_GTK2_did_we_find_everything false)
    endif()
endforeach()

if(GTK2_USE_IMPORTED_TARGETS)
    set(GTK2_LIBRARIES ${GTK2_TARGETS})
endif()


if(_GTK2_did_we_find_everything AND NOT GTK2_VERSION_CHECK_FAILED)
    set(GTK2_FOUND true)
else()
    # Unset our variables.
    set(GTK2_FOUND false)
    set(GTK2_VERSION)
    set(GTK2_VERSION_MAJOR)
    set(GTK2_VERSION_MINOR)
    set(GTK2_VERSION_PATCH)
    set(GTK2_INCLUDE_DIRS)
    set(GTK2_LIBRARIES)
    set(GTK2_TARGETS)
    set(GTK2_DEFINITIONS)
endif()

if(GTK2_INCLUDE_DIRS)
  list(REMOVE_DUPLICATES GTK2_INCLUDE_DIRS)
endif()
