# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGTK
-------

.. note::

  This module works only on Unix-like systems and was intended for early GTK
  branch of 1.x, which is no longer maintained.  Use the latest supported GTK
  version and :module:`FindPkgConfig` module to find GTK in CMake instead of
  this module.  For example:

  .. code-block:: cmake

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK REQUIRED IMPORTED_TARGET gtk4>=4.14)
    target_link_libraries(example PRIVATE PkgConfig::GTK)

Finds GTK, glib and GTKGLArea:

.. code-block:: cmake

  find_package(GTK [...])

GTK is a multi-platform toolkit for creating graphical user interfaces.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``GTK_FOUND``
  Boolean indicating whether GTK was found.
``GTK_GL_FOUND``
  Boolean indicating whether GTK's GL features were found.
``GTK_INCLUDE_DIR``
  Include directories containing headers needed to use GTK.
``GTK_LIBRARIES``
  Libraries needed to link against for using GTK.

Examples
^^^^^^^^

Finding GTK 1.x and creating an interface :ref:`imported target
<Imported Targets>` that encapsulates its usage requirements for linking to a
project target:

.. code-block:: cmake

  find_package(GTK)

  if(GTK_FOUND)
    add_library(GTK::GTK INTERFACE IMPORTED)
    set_target_properties(
      GTK::GTK
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GTK_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${GTK_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE GTK::GTK)

See Also
^^^^^^^^

* The :module:`FindGTK2` module to find GTK version 2.
#]=======================================================================]

# don't even bother under WIN32
if(UNIX)

  find_path( GTK_gtk_INCLUDE_PATH NAMES gtk/gtk.h
    PATH_SUFFIXES gtk-1.2 gtk12
    PATHS
    /usr/openwin/share/include
    /usr/openwin/include
    /opt/gnome/include
  )

  # Some Linux distributions (e.g. Red Hat) have glibconfig.h
  # and glib.h in different directories, so we need to look
  # for both.
  #  - Atanas Georgiev <atanas@cs.columbia.edu>

  find_path( GTK_glibconfig_INCLUDE_PATH NAMES glibconfig.h
    PATH_SUFFIXES glib/include lib/glib/include include/glib12
    PATHS
    /usr/openwin/share/include
    /opt/gnome/include
    /opt/gnome/lib/glib/include
  )

  find_path( GTK_glib_INCLUDE_PATH NAMES glib.h
    PATH_SUFFIXES gtk-1.2 glib-1.2 glib12 glib/include lib/glib/include
    PATHS
    /usr/openwin/share/include
    /opt/gnome/include
  )

  find_path( GTK_gtkgl_INCLUDE_PATH NAMES gtkgl/gtkglarea.h
    PATHS /usr/openwin/share/include
          /opt/gnome/include
  )

  find_library( GTK_gtkgl_LIBRARY gtkgl
    /usr/openwin/lib
    /opt/gnome/lib
  )

  #
  # The 12 suffix is thanks to the FreeBSD ports collection
  #

  find_library( GTK_gtk_LIBRARY
    NAMES  gtk gtk12
    PATHS /usr/openwin/lib
          /opt/gnome/lib
  )

  find_library( GTK_gdk_LIBRARY
    NAMES  gdk gdk12
    PATHS  /usr/openwin/lib
           /opt/gnome/lib
  )

  find_library( GTK_gmodule_LIBRARY
    NAMES  gmodule gmodule12
    PATHS  /usr/openwin/lib
           /opt/gnome/lib
  )

  find_library( GTK_glib_LIBRARY
    NAMES  glib glib12
    PATHS  /usr/openwin/lib
           /opt/gnome/lib
  )

  find_library( GTK_Xi_LIBRARY
    NAMES Xi
    PATHS /usr/openwin/lib
          /opt/gnome/lib
    )

  find_library( GTK_gthread_LIBRARY
    NAMES  gthread gthread12
    PATHS  /usr/openwin/lib
           /opt/gnome/lib
  )

  if(GTK_gtk_INCLUDE_PATH
     AND GTK_glibconfig_INCLUDE_PATH
     AND GTK_glib_INCLUDE_PATH
     AND GTK_gtk_LIBRARY
     AND GTK_glib_LIBRARY)

    # Assume that if gtk and glib were found, the other
    # supporting libraries have also been found.

    set( GTK_FOUND "YES" )
    set( GTK_INCLUDE_DIR  ${GTK_gtk_INCLUDE_PATH}
                           ${GTK_glibconfig_INCLUDE_PATH}
                           ${GTK_glib_INCLUDE_PATH} )
    set( GTK_LIBRARIES  ${GTK_gtk_LIBRARY}
                        ${GTK_gdk_LIBRARY}
                        ${GTK_glib_LIBRARY} )

    if(GTK_gmodule_LIBRARY)
      set(GTK_LIBRARIES ${GTK_LIBRARIES} ${GTK_gmodule_LIBRARY})
    endif()
    if(GTK_gthread_LIBRARY)
      set(GTK_LIBRARIES ${GTK_LIBRARIES} ${GTK_gthread_LIBRARY})
    endif()
    if(GTK_Xi_LIBRARY)
      set(GTK_LIBRARIES ${GTK_LIBRARIES} ${GTK_Xi_LIBRARY})
    endif()

    if(GTK_gtkgl_INCLUDE_PATH AND GTK_gtkgl_LIBRARY)
      set( GTK_GL_FOUND "YES" )
      set( GTK_INCLUDE_DIR  ${GTK_INCLUDE_DIR}
                            ${GTK_gtkgl_INCLUDE_PATH} )
      set( GTK_LIBRARIES  ${GTK_gtkgl_LIBRARY} ${GTK_LIBRARIES} )
      mark_as_advanced(
        GTK_gtkgl_LIBRARY
        GTK_gtkgl_INCLUDE_PATH
        )
    endif()

  endif()

  mark_as_advanced(
    GTK_gdk_LIBRARY
    GTK_glib_INCLUDE_PATH
    GTK_glib_LIBRARY
    GTK_glibconfig_INCLUDE_PATH
    GTK_gmodule_LIBRARY
    GTK_gthread_LIBRARY
    GTK_Xi_LIBRARY
    GTK_gtk_INCLUDE_PATH
    GTK_gtk_LIBRARY
    GTK_gtkgl_INCLUDE_PATH
    GTK_gtkgl_LIBRARY
  )

endif()
