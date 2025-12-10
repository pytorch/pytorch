# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
UseQt4
------

Use Module for QT4

Sets up C and C++ to use Qt 4.  It is assumed that :module:`FindQt` has
already been loaded.  See :module:`FindQt` for information on how to load
Qt 4 into your CMake project.
#]=======================================================================]

add_definitions(${QT_DEFINITIONS})
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<NOT:$<CONFIG:Debug>>:QT_NO_DEBUG>)

if(QT_INCLUDE_DIRS_NO_SYSTEM)
  include_directories(${QT_INCLUDE_DIR})
else()
  include_directories(SYSTEM ${QT_INCLUDE_DIR})
endif()

set(QT_LIBRARIES "")
set(QT_LIBRARIES_PLUGINS "")

if (QT_USE_QTMAIN)
  if (Q_WS_WIN)
    set(QT_LIBRARIES ${QT_LIBRARIES} ${QT_QTMAIN_LIBRARY})
  endif ()
endif ()

if(QT_DONT_USE_QTGUI)
  set(QT_USE_QTGUI 0)
else()
  set(QT_USE_QTGUI 1)
endif()

if(QT_DONT_USE_QTCORE)
  set(QT_USE_QTCORE 0)
else()
  set(QT_USE_QTCORE 1)
endif()

if (QT_USE_QT3SUPPORT)
  add_definitions(-DQT3_SUPPORT)
endif ()

# list dependent modules, so dependent libraries are added
set(QT_QT3SUPPORT_MODULE_DEPENDS QTGUI QTSQL QTXML QTNETWORK QTCORE)
set(QT_QTSVG_MODULE_DEPENDS QTGUI QTCORE)
set(QT_QTUITOOLS_MODULE_DEPENDS QTGUI QTXML QTCORE)
set(QT_QTHELP_MODULE_DEPENDS QTGUI QTSQL QTXML QTNETWORK QTCORE)
if(QT_QTDBUS_FOUND)
  set(QT_PHONON_MODULE_DEPENDS QTGUI QTDBUS QTCORE)
else()
  set(QT_PHONON_MODULE_DEPENDS QTGUI QTCORE)
endif()
set(QT_QTDBUS_MODULE_DEPENDS QTXML QTCORE)
set(QT_QTXMLPATTERNS_MODULE_DEPENDS QTNETWORK QTCORE)
set(QT_QAXCONTAINER_MODULE_DEPENDS QTGUI QTCORE)
set(QT_QAXSERVER_MODULE_DEPENDS QTGUI QTCORE)
set(QT_QTSCRIPTTOOLS_MODULE_DEPENDS QTGUI QTCORE)
set(QT_QTWEBKIT_MODULE_DEPENDS QTXMLPATTERNS QTGUI QTCORE)
set(QT_QTDECLARATIVE_MODULE_DEPENDS QTSCRIPT QTSVG QTSQL QTXMLPATTERNS QTGUI QTCORE)
set(QT_QTMULTIMEDIA_MODULE_DEPENDS QTGUI QTCORE)
set(QT_QTOPENGL_MODULE_DEPENDS QTGUI QTCORE)
set(QT_QTSCRIPT_MODULE_DEPENDS QTCORE)
set(QT_QTGUI_MODULE_DEPENDS QTCORE)
set(QT_QTTEST_MODULE_DEPENDS QTCORE)
set(QT_QTXML_MODULE_DEPENDS QTCORE)
set(QT_QTSQL_MODULE_DEPENDS QTCORE)
set(QT_QTNETWORK_MODULE_DEPENDS QTCORE)

# Qt modules  (in order of dependence)
foreach(module QT3SUPPORT QTOPENGL QTASSISTANT QTDESIGNER QTMOTIF QTNSPLUGIN
               QAXSERVER QAXCONTAINER QTDECLARATIVE QTSCRIPT QTSVG QTUITOOLS QTHELP
               QTWEBKIT PHONON QTSCRIPTTOOLS QTMULTIMEDIA QTXMLPATTERNS QTGUI QTTEST
               QTDBUS QTXML QTSQL QTNETWORK QTCORE)

  if (QT_USE_${module} OR QT_USE_${module}_DEPENDS)
    if (QT_${module}_FOUND)
      if(QT_USE_${module})
        string(REPLACE "QT" "" qt_module_def "${module}")
        add_definitions(-DQT_${qt_module_def}_LIB)
        if(QT_INCLUDE_DIRS_NO_SYSTEM)
          include_directories(${QT_${module}_INCLUDE_DIR})
        else()
          include_directories(SYSTEM ${QT_${module}_INCLUDE_DIR})
        endif()
      endif()
      if(QT_USE_${module} OR QT_IS_STATIC)
        set(QT_LIBRARIES ${QT_LIBRARIES} ${QT_${module}_LIBRARY})
      endif()
      set(QT_LIBRARIES_PLUGINS ${QT_LIBRARIES_PLUGINS} ${QT_${module}_PLUGINS})
      if(QT_IS_STATIC)
        set(QT_LIBRARIES ${QT_LIBRARIES} ${QT_${module}_LIB_DEPENDENCIES})
      endif()
      foreach(depend_module ${QT_${module}_MODULE_DEPENDS})
        set(QT_USE_${depend_module}_DEPENDS 1)
      endforeach()
    else ()
      message("Qt ${module} library not found.")
    endif ()
  endif ()

endforeach()
