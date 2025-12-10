AUTOMOC
-------

Should the target be processed with auto-moc (for Qt projects).

``AUTOMOC`` is a boolean specifying whether CMake will handle the Qt ``moc``
preprocessor automatically, i.e.  without having to use commands like
:module:`qt4_wrap_cpp() <FindQt4>`, `qt5_wrap_cpp()`_, etc.
Currently, Qt versions 4 to 6 are supported.

.. _qt5_wrap_cpp(): https://doc.qt.io/archives/qt-5.15/qtcore-cmake-qt5-wrap-cpp.html

This property is initialized by the value of the :variable:`CMAKE_AUTOMOC`
variable if it is set when a target is created.

When this property is set ``ON``, CMake will scan the header and
source files at build time and invoke ``moc`` accordingly.


Header file processing
^^^^^^^^^^^^^^^^^^^^^^

At configuration time, a list of header files that should be scanned by
``AUTOMOC`` is computed from the target's sources.

- All header files in the target's sources are added to the scan list.
- For all C++ source files ``<source_base>.<source_extension>`` in the
  target's sources, CMake searches for

  - a regular header with the same base name
    (``<source_base>.<header_extension>``) and
  - a private header with the same base name and a ``_p`` suffix
    (``<source_base>_p.<header_extension>``)

  and adds these to the scan list.

- The :prop_tgt:`AUTOMOC_INCLUDE_DIRECTORIES` target property may be set
  to explicitly tell ``moc`` what include directories to search.  If not
  set, the default is to use the include directories from the target and
  its transitive closure of dependencies.

At build time, CMake scans each unknown or modified header file from the
list and searches for

- a Qt macro from :prop_tgt:`AUTOMOC_MACRO_NAMES`,
- additional file dependencies from the ``FILE`` argument of a
  ``Q_PLUGIN_METADATA`` macro and
- additional file dependencies detected by filters defined in
  :prop_tgt:`AUTOMOC_DEPEND_FILTERS`.

If a Qt macro is found, then the header will be compiled by the ``moc`` to the
output file ``moc_<base_name>.cpp``.  The complete output file path is
described in the section `Output file location`_.

The header will be ``moc`` compiled again if a file from the additional file
dependencies changes.

Header ``moc`` output files ``moc_<base_name>.cpp`` can be included in source
files.  In the section `Including header moc files in sources`_ there is more
information on that topic.


Source file processing
^^^^^^^^^^^^^^^^^^^^^^

At build time, CMake scans each unknown or modified C++ source file from the
target's sources for

- a Qt macro from :prop_tgt:`AUTOMOC_MACRO_NAMES`,
- includes of header ``moc`` files
  (see `Including header moc files in sources`_),
- additional file dependencies from the ``FILE`` argument of a
  ``Q_PLUGIN_METADATA`` macro and
- additional file dependencies detected by filters defined in
  :prop_tgt:`AUTOMOC_DEPEND_FILTERS`.

If a Qt macro is found, then the C++ source file
``<base>.<source_extension>`` is expected to as well contain an include
statement

.. code-block:: c++

  #include <<base>.moc> // or
  #include "<base>.moc"

The source file then will be compiled by the ``moc`` to the output file
``<base>.moc``.  A description of the complete output file path is in section
`Output file location`_.

The source will be ``moc`` compiled again if a file from the additional file
dependencies changes.

Including header moc files in sources
"""""""""""""""""""""""""""""""""""""

A source file can include the ``moc`` output file of a header
``<header_base>.<header_extension>`` by using an include statement of
the form

.. code-block:: c++

  #include <moc_<header_base>.cpp> // or
  #include "moc_<header_base>.cpp"

If the ``moc`` output file of a header is included by a source, it will
be generated in a different location than if it was not included.  This is
described in the section `Output file location`_.


Output file location
^^^^^^^^^^^^^^^^^^^^

Included moc output files
"""""""""""""""""""""""""

``moc`` output files that are included by a source file will be generated in

- ``<AUTOGEN_BUILD_DIR>/include``
  for single configuration generators or in
- ``<AUTOGEN_BUILD_DIR>/include_<CONFIG>``
  for :prop_gbl:`multi configuration <GENERATOR_IS_MULTI_CONFIG>` generators.

Where ``<AUTOGEN_BUILD_DIR>`` is the value of the target property
:prop_tgt:`AUTOGEN_BUILD_DIR`.

The include directory is automatically added to the target's
:prop_tgt:`INCLUDE_DIRECTORIES`.

Not included moc output files
"""""""""""""""""""""""""""""

``moc`` output files that are not included in a source file will be generated
in

- ``<AUTOGEN_BUILD_DIR>/<SOURCE_DIR_CHECKSUM>``
  for single configuration generators or in,
- ``<AUTOGEN_BUILD_DIR>/include_<CONFIG>/<SOURCE_DIR_CHECKSUM>``
  for :prop_gbl:`multi configuration <GENERATOR_IS_MULTI_CONFIG>` generators.

Where ``<SOURCE_DIR_CHECKSUM>`` is a checksum computed from the relative
parent directory path of the ``moc`` input file.  This scheme allows to have
``moc`` input files with the same name in different directories.

All not included ``moc`` output files will be included automatically by the
CMake generated file

- ``<AUTOGEN_BUILD_DIR>/mocs_compilation.cpp``, or
- ``<AUTOGEN_BUILD_DIR>/mocs_compilation_$<CONFIG>.cpp``,

which is added to the target's sources.


Qt version detection
^^^^^^^^^^^^^^^^^^^^

``AUTOMOC`` enabled targets need to know the Qt major and minor
version they're working with.  The major version usually is provided by the
``INTERFACE_QT_MAJOR_VERSION`` property of the ``Qt[456]Core`` library,
that the target links to.  To find the minor version, CMake builds a list of
available Qt versions from

- ``Qt6Core_VERSION_MAJOR`` and ``Qt6Core_VERSION_MINOR`` variables
  (usually set by ``find_package(Qt6...)``)
- ``Qt6Core_VERSION_MAJOR`` and ``Qt6Core_VERSION_MINOR`` directory properties
- ``Qt5Core_VERSION_MAJOR`` and ``Qt5Core_VERSION_MINOR`` variables
  (usually set by ``find_package(Qt5...)``)
- ``Qt5Core_VERSION_MAJOR`` and ``Qt5Core_VERSION_MINOR`` directory properties
- ``QT_VERSION_MAJOR`` and ``QT_VERSION_MINOR``  variables
  (usually set by ``find_package(Qt4...)``)
- ``QT_VERSION_MAJOR`` and ``QT_VERSION_MINOR``  directory properties

in the context of the :command:`add_executable` or :command:`add_library` call.

Assumed  ``INTERFACE_QT_MAJOR_VERSION`` is a valid number, the first
entry in the list with a matching major version is taken.  If no matching major
version was found, an error is generated.
If  ``INTERFACE_QT_MAJOR_VERSION`` is not a valid number, the first
entry in the list is taken.

A ``find_package(Qt[456]...)`` call sets the ``QT/Qt[56]Core_VERSION_MAJOR/MINOR``
variables.  If the call is in a different context than the
:command:`add_executable` or :command:`add_library` call, e.g. in a function,
then the version variables might not be available to the ``AUTOMOC``
enabled target.
In that case the version variables can be forwarded from the
``find_package(Qt[456]...)`` calling context to the :command:`add_executable`
or :command:`add_library` calling context as directory properties.
The following Qt5 example demonstrates the procedure.

.. code-block:: cmake

  function (add_qt5_client)
    find_package(Qt5 REQUIRED QUIET COMPONENTS Core Widgets)
    ...
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      PROPERTY Qt5Core_VERSION_MAJOR "${Qt5Core_VERSION_MAJOR}")
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      PROPERTY Qt5Core_VERSION_MINOR "${Qt5Core_VERSION_MAJOR}")
    ...
  endfunction ()
  ...
  add_qt5_client()
  add_executable(myTarget main.cpp)
  target_link_libraries(myTarget Qt5::QtWidgets)
  set_property(TARGET myTarget PROPERTY AUTOMOC ON)


Modifiers
^^^^^^^^^

:prop_tgt:`AUTOMOC_EXECUTABLE`:
The ``moc`` executable will be detected automatically, but can be forced to
a certain binary using this target property.

:prop_tgt:`AUTOMOC_MOC_OPTIONS`:
Additional command line options for ``moc`` can be set in this target property.

:prop_tgt:`AUTOMOC_MACRO_NAMES`:
This list of Qt macro names can be extended to search for additional macros in
headers and sources.

:prop_tgt:`AUTOMOC_DEPEND_FILTERS`:
``moc`` dependency file names can be extracted from headers or sources by
defining file name filters in this target property.

:prop_tgt:`AUTOMOC_COMPILER_PREDEFINES`:
Compiler pre definitions for ``moc`` are written to the ``moc_predefs.h`` file.
The generation of this file can be enabled or disabled in this target property.

:prop_tgt:`AUTOMOC_INCLUDE_DIRECTORIES`:
Specifies one or more include directories for ``AUTOMOC`` to pass explicitly to ``moc``
instead of automatically discovering a target’s include directories.

:prop_sf:`SKIP_AUTOMOC`:
Sources and headers can be excluded from ``AUTOMOC`` processing by
setting this source file property.

:prop_sf:`SKIP_AUTOGEN`:
Source files can be excluded from ``AUTOMOC``,
:prop_tgt:`AUTOUIC` and :prop_tgt:`AUTORCC` processing by
setting this source file property.

:prop_gbl:`AUTOGEN_SOURCE_GROUP`:
This global property can be used to group files generated by
``AUTOMOC`` or :prop_tgt:`AUTORCC` together in an IDE, e.g.  in MSVS.

:prop_gbl:`AUTOGEN_TARGETS_FOLDER`:
This global property can be used to group ``AUTOMOC``,
:prop_tgt:`AUTOUIC` and :prop_tgt:`AUTORCC` targets together in an IDE,
e.g.  in MSVS.

:variable:`CMAKE_GLOBAL_AUTOGEN_TARGET`:
A global ``autogen`` target, that depends on all ``AUTOMOC`` or
:prop_tgt:`AUTOUIC` generated :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>`
targets in the project, will be generated when this variable is ``ON``.

:prop_tgt:`AUTOGEN_PARALLEL`:
This target property controls the number of ``moc`` or ``uic`` processes to
start in parallel during builds.

:prop_tgt:`AUTOGEN_COMMAND_LINE_LENGTH_MAX`:
This target property controls the limit when to use response files for
``moc`` or ``uic`` processes on Windows.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.
