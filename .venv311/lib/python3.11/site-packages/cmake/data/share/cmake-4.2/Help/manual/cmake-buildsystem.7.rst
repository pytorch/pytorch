.. cmake-manual-description: CMake Buildsystem Reference

cmake-buildsystem(7)
********************

.. only:: html

   .. contents::

Introduction
============

A CMake-based buildsystem is organized as a set of high-level logical
targets.  Each target corresponds to an executable or library, or
is a custom target containing custom commands.  Dependencies between the
targets are expressed in the buildsystem to determine the build order
and the rules for regeneration in response to change.

Binary Targets
==============

Executables and libraries are defined using the :command:`add_executable`
and :command:`add_library` commands.  The resulting binary files have
appropriate :prop_tgt:`PREFIX`, :prop_tgt:`SUFFIX` and extensions for the
platform targeted. Dependencies between binary targets are expressed using
the :command:`target_link_libraries` command:

.. code-block:: cmake

  add_library(archive archive.cpp zip.cpp lzma.cpp)
  add_executable(zipapp zipapp.cpp)
  target_link_libraries(zipapp archive)

``archive`` is defined as a ``STATIC`` library -- an archive containing objects
compiled from ``archive.cpp``, ``zip.cpp``, and ``lzma.cpp``.  ``zipapp``
is defined as an executable formed by compiling and linking ``zipapp.cpp``.
When linking the ``zipapp`` executable, the ``archive`` static library is
linked in.

.. _`Executables`:

Executables
-----------

Executables are binaries created by linking object files together,
one of which contains a program entry point, e.g., ``main``.

The :command:`add_executable` command defines an executable target:

.. code-block:: cmake

  add_executable(mytool mytool.cpp)

CMake generates build rules to compile the source files into object
files and link them into an executable.

Link dependencies of executables may be specified using the
:command:`target_link_libraries` command.  Linkers start with the
object files compiled from the executable's own source files, and
then resolve remaining symbol dependencies by searching linked libraries.

Commands such as :command:`add_custom_command`, which generates rules to be
run at build time can transparently use an :prop_tgt:`EXECUTABLE <TYPE>`
target as a ``COMMAND`` executable.  The buildsystem rules will ensure that
the executable is built before attempting to run the command.

.. _`Static Libraries`:

Static Libraries
----------------

Static libraries are archives of object files.  They are produced by an
archiver, not a linker.  `Executables`_, `Shared Libraries`_, and
`Module Libraries`_ may link to static libraries as dependencies.
Linkers select subsets of object files from static libraries as needed
to resolve symbols and link them into consuming binaries.  Each binary
that links to a static library gets its own copy of the symbols, and
the static library itself is not needed at runtime.

The :command:`add_library` command defines a static library target
when called with the ``STATIC`` library type:

.. code-block:: cmake

  add_library(archive STATIC archive.cpp zip.cpp lzma.cpp)

or, when the :variable:`BUILD_SHARED_LIBS` variable is false, with no type:

.. code-block:: cmake

  add_library(archive archive.cpp zip.cpp lzma.cpp)

CMake generates build rules to compile the source files into object
files and archive them into a static library.

Link dependencies of static libraries may be specified using the
:command:`target_link_libraries` command.  Since static libraries are
archives rather than linked binaries, object files from their link
dependencies are not included in the libraries themselves (except for
`Object Libraries`_ specified as *direct* link dependencies).
Instead, CMake records static libraries' link dependencies for
transitive use when linking consuming binaries.

.. _`Shared Libraries`:

Shared Libraries
----------------

Shared libraries are binaries created by linking object files together.
`Executables`_, other shared libraries, and `Module Libraries`_ may link
to shared libraries as dependencies.  Linkers record references to shared
libraries in consuming binaries.  At runtime, a dynamic loader searches
for referenced shared libraries on disk and loads their symbols.

The :command:`add_library` command defines a shared library target
when called with the ``SHARED`` library type:

.. code-block:: cmake

  add_library(archive SHARED archive.cpp zip.cpp lzma.cpp)

or, when the :variable:`BUILD_SHARED_LIBS` variable is true, with no type:

.. code-block:: cmake

  add_library(archive archive.cpp zip.cpp lzma.cpp)

CMake generates build rules to compile the source files into object
files and link them into a shared library.

Link dependencies of shared libraries may be specified using the
:command:`target_link_libraries` command.  Linkers start with the
object files compiled from the shared library's own source files, and
then resolve remaining symbol dependencies by searching linked libraries.

.. note::

  CMake expects shared libraries to export at least one symbol.  If a library
  does not export any unmanaged symbols, e.g., a Windows resource DLL or
  C++/CLI DLL, make it a `Module Library <Module Libraries_>`_ instead.

.. _`Apple Frameworks`:

Apple Frameworks
----------------

`Shared Libraries`_ and `Static Libraries`_ may be marked with the
:prop_tgt:`FRAMEWORK` target property to create a macOS or iOS Framework.
A library with the ``FRAMEWORK`` target property should also set the
:prop_tgt:`FRAMEWORK_VERSION` target property.  This property is typically
set to the value of "A" by macOS conventions.
The ``MACOSX_FRAMEWORK_IDENTIFIER`` sets the ``CFBundleIdentifier`` key
and it uniquely identifies the bundle.

.. code-block:: cmake

  add_library(MyFramework SHARED MyFramework.cpp)
  set_target_properties(MyFramework PROPERTIES
    FRAMEWORK TRUE
    FRAMEWORK_VERSION A # Version "A" is macOS convention
    MACOSX_FRAMEWORK_IDENTIFIER org.cmake.MyFramework
  )

.. _`Module Libraries`:

Module Libraries
----------------

Module libraries are binaries created by linking object files together.
Unlike `Shared Libraries`_, module libraries may not be linked by other
binaries as dependencies -- do not name them in the right-hand side of
the :command:`target_link_libraries` command.  Instead, module libraries
are plugins that an application can dynamically load on-demand at runtime,
e.g., by ``dlopen``.

The :command:`add_library` command defines a module library target
when called with the ``MODULE`` library type:

.. code-block:: cmake

  add_library(archivePlugin MODULE 7z.cpp)

CMake generates build rules to compile the source files into object
files and link them into a module library.

Link dependencies of module libraries may be specified using the
:command:`target_link_libraries` command.  Linkers start with the
object files compiled from the module library's own source files, and
then resolve remaining symbol dependencies by searching linked libraries.

.. _`Object Libraries`:

Object Libraries
----------------

Object libraries are collections of object files created by compiling
source files without any archiving or linking.  The object files may be
used when linking `Executables`_, `Shared Libraries`_, and
`Module Libraries`_, or when archiving `Static Libraries`_.

The :command:`add_library` command defines an object library target
when called with the ``OBJECT`` library type:

.. code-block:: cmake

  add_library(archiveObjs OBJECT archive.cpp zip.cpp lzma.cpp)

CMake generates build rules to compile the source files into object files.

Other targets may specify the object files as source inputs by using the
:manual:`generator expression <cmake-generator-expressions(7)>` syntax
:genex:`$<TARGET_OBJECTS:name>`:

.. code-block:: cmake

  add_library(archiveExtras STATIC $<TARGET_OBJECTS:archiveObjs> extras.cpp)

  add_executable(test_exe $<TARGET_OBJECTS:archiveObjs> test.cpp)

The consuming targets are linked (or archived) using object files
both from their own sources and from the named object libraries.

Alternatively, object libraries may be specified as link dependencies
of other targets:

.. code-block:: cmake

  add_library(archiveExtras STATIC extras.cpp)
  target_link_libraries(archiveExtras PUBLIC archiveObjs)

  add_executable(test_exe test.cpp)
  target_link_libraries(test_exe archiveObjs)

The consuming targets are linked (or archived) using object files
both from their own sources and from object libraries specified as
*direct* link dependencies by :command:`target_link_libraries`.
See :ref:`Linking Object Libraries`.

Object libraries may not be used as the ``TARGET`` in a use of the
:command:`add_custom_command(TARGET)` command signature.  However,
the list of objects can be used by :command:`add_custom_command(OUTPUT)`
or :command:`file(GENERATE)` by using ``$<TARGET_OBJECTS:objlib>``.

Build Specification and Usage Requirements
==========================================

Targets build according to their own
`build specification <Target Build Specification_>`_ in combination with
`usage requirements <Target Usage Requirements_>`_ propagated from their
link dependencies.  Both may be specified using target-specific
`commands <Target Commands_>`_.

For example:

.. code-block:: cmake

  add_library(archive SHARED archive.cpp zip.cpp)

  if (LZMA_FOUND)
    # Add a source implementing support for lzma.
    target_sources(archive PRIVATE lzma.cpp)

    # Compile the 'archive' library sources with '-DBUILDING_WITH_LZMA'.
    target_compile_definitions(archive PRIVATE BUILDING_WITH_LZMA)
  endif()

  target_compile_definitions(archive INTERFACE USING_ARCHIVE_LIB)

  add_executable(consumer consumer.cpp)

  # Link 'consumer' to 'archive'.  This also consumes its usage requirements,
  # so 'consumer.cpp' is compiled with '-DUSING_ARCHIVE_LIB'.
  target_link_libraries(consumer archive)


Target Commands
---------------

Target-specific commands populate the
`build specification <Target Build Specification_>`_ of `Binary Targets`_ and
`usage requirements <Target Usage Requirements_>`_ of `Binary Targets`_,
`Interface Libraries`_, and `Imported Targets`_.

.. _`Target Command Scope`:

Invocations must specify scope keywords, each affecting the visibility
of arguments following it.  The scopes are:

``PUBLIC``
  Populates both properties for `building <Target Build Specification_>`_
  and properties for `using <Target Usage Requirements_>`_ a target.

``PRIVATE``
  Populates only properties for `building <Target Build Specification_>`_
  a target.

``INTERFACE``
  Populates only properties for `using <Target Usage Requirements_>`_
  a target.

The commands are:

:command:`target_compile_definitions`
  Populates the :prop_tgt:`COMPILE_DEFINITIONS` build specification and
  :prop_tgt:`INTERFACE_COMPILE_DEFINITIONS` usage requirement properties.

  For example, the call

  .. code-block:: cmake

    target_compile_definitions(archive
      PRIVATE   BUILDING_WITH_LZMA
      INTERFACE USING_ARCHIVE_LIB
    )

  appends ``BUILDING_WITH_LZMA`` to the target's ``COMPILE_DEFINITIONS``
  property and appends ``USING_ARCHIVE_LIB`` to the target's
  ``INTERFACE_COMPILE_DEFINITIONS`` property.

:command:`target_compile_options`
  Populates the :prop_tgt:`COMPILE_OPTIONS` build specification and
  :prop_tgt:`INTERFACE_COMPILE_OPTIONS` usage requirement properties.

:command:`target_compile_features`
  .. versionadded:: 3.1

  Populates the :prop_tgt:`COMPILE_FEATURES` build specification and
  :prop_tgt:`INTERFACE_COMPILE_FEATURES` usage requirement properties.

:command:`target_include_directories`
  Populates the :prop_tgt:`INCLUDE_DIRECTORIES` build specification
  and :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` usage requirement
  properties.  With the ``SYSTEM`` option, it also populates the
  :prop_tgt:`INTERFACE_SYSTEM_INCLUDE_DIRECTORIES` usage requirement.

  For convenience, the :variable:`CMAKE_INCLUDE_CURRENT_DIR` variable
  may be enabled to add the source directory and corresponding build
  directory as ``INCLUDE_DIRECTORIES`` on all targets.  Similarly,
  the :variable:`CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE` variable may
  be enabled to add them as ``INTERFACE_INCLUDE_DIRECTORIES`` on all
  targets.

:command:`target_sources`
  .. versionadded:: 3.1

  Populates the :prop_tgt:`SOURCES` build specification and
  :prop_tgt:`INTERFACE_SOURCES` usage requirement properties.

  It also supports specifying :ref:`File Sets`, which can add C++ module
  sources and headers not listed in the ``SOURCES`` and ``INTERFACE_SOURCES``
  properties.  File sets may also populate the :prop_tgt:`INCLUDE_DIRECTORIES`
  build specification and :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` usage
  requirement properties with the include directories containing the headers.

:command:`target_precompile_headers`
  .. versionadded:: 3.16

  Populates the :prop_tgt:`PRECOMPILE_HEADERS` build specification and
  :prop_tgt:`INTERFACE_PRECOMPILE_HEADERS` usage requirement properties.

:command:`target_link_libraries`
  Populates the :prop_tgt:`LINK_LIBRARIES` build specification
  and :prop_tgt:`INTERFACE_LINK_LIBRARIES` usage requirement properties.

  This is the primary mechanism by which link dependencies and their
  `usage requirements <Target Usage Requirements_>`_ are transitively
  propagated to affect compilation and linking of a target.

:command:`target_link_directories`
  .. versionadded:: 3.13

  Populates the :prop_tgt:`LINK_DIRECTORIES` build specification and
  :prop_tgt:`INTERFACE_LINK_DIRECTORIES` usage requirement properties.

:command:`target_link_options`
  .. versionadded:: 3.13

  Populates the :prop_tgt:`LINK_OPTIONS` build specification and
  :prop_tgt:`INTERFACE_LINK_OPTIONS` usage requirement properties.

.. _`Target Build Specification`:

Target Build Specification
--------------------------

The build specification of `Binary Targets`_ is represented by target
properties.  For each of the following `compile <Target Compile Properties_>`_
and `link <Target Link Properties_>`_ properties, compilation and linking
of the target is affected both by its own value and by the corresponding
`usage requirement <Target Usage Requirements_>`_ property, named with
an ``INTERFACE_`` prefix, collected from the transitive closure of link
dependencies.

.. _`Target Compile Properties`:

Target Compile Properties
^^^^^^^^^^^^^^^^^^^^^^^^^

These represent the `build specification <Target Build Specification_>`_
for compiling a target.

:prop_tgt:`COMPILE_DEFINITIONS`
  List of compile definitions for compiling sources in the target.
  These are passed to the compiler with ``-D`` flags, or equivalent,
  in an unspecified order.

  The :prop_tgt:`DEFINE_SYMBOL` target property is also used
  as a compile definition as a special convenience case for
  ``SHARED`` and ``MODULE`` library targets.

:prop_tgt:`COMPILE_OPTIONS`
  List of compile options for compiling sources in the target.
  These are passed to the compiler as flags, in the order of appearance.

  Compile options are automatically escaped for the shell.

  Some compile options are best specified via dedicated settings,
  such as the :prop_tgt:`POSITION_INDEPENDENT_CODE` target property.

:prop_tgt:`COMPILE_FEATURES`
  .. versionadded:: 3.1

  List of :manual:`compile features <cmake-compile-features(7)>` needed
  for compiling sources in the target.  Typically these ensure the
  target's sources are compiled using a sufficient language standard level.

:prop_tgt:`INCLUDE_DIRECTORIES`
  List of include directories for compiling sources in the target.
  These are passed to the compiler with ``-I`` or ``-isystem`` flags,
  or equivalent, in the order of appearance.

  For convenience, the :variable:`CMAKE_INCLUDE_CURRENT_DIR` variable
  may be enabled to add the source directory and corresponding build
  directory as ``INCLUDE_DIRECTORIES`` on all targets.

:prop_tgt:`SOURCES`
  List of source files associated with the target.  This includes sources
  specified when the target was created by the :command:`add_executable`,
  :command:`add_library`, or :command:`add_custom_target` command.
  It also includes sources added by the :command:`target_sources` command,
  but does not include :ref:`File Sets`.

:prop_tgt:`PRECOMPILE_HEADERS`
  .. versionadded:: 3.16

  List of header files to precompile and include when compiling
  sources in the target.

:prop_tgt:`AUTOMOC_MACRO_NAMES`
  .. versionadded:: 3.10

  List of macro names used by :prop_tgt:`AUTOMOC` to determine if a
  C++ source in the target needs to be processed by ``moc``.

:prop_tgt:`AUTOUIC_OPTIONS`
  .. versionadded:: 3.0

  List of options used by :prop_tgt:`AUTOUIC` when invoking ``uic``
  for the target.

.. _`Target Link Properties`:

Target Link Properties
^^^^^^^^^^^^^^^^^^^^^^

These represent the `build specification <Target Build Specification_>`_
for linking a target.

:prop_tgt:`LINK_LIBRARIES`
  List of link libraries for linking the target, if it is an executable,
  shared library, or module library.  Entries for `Static Libraries`_
  and `Shared Libraries`_ are passed to the linker either via paths to
  their link artifacts, or with ``-l`` flags or equivalent.  Entries for
  `Object Libraries`_ are passed to the linker via paths to their object
  files.

  Additionally, for compiling and linking the target itself,
  `usage requirements <Target Usage Requirements_>`_ are propagated from
  ``LINK_LIBRARIES`` entries naming `Static Libraries`_, `Shared Libraries`_,
  `Interface Libraries`_, `Object Libraries`_, and `Imported Targets`_,
  collected over the transitive closure of their
  :prop_tgt:`INTERFACE_LINK_LIBRARIES` properties.

:prop_tgt:`LINK_DIRECTORIES`
  .. versionadded:: 3.13

  List of link directories for linking the target, if it is an executable,
  shared library, or module library.  The directories are passed to the
  linker with ``-L`` flags, or equivalent.

:prop_tgt:`LINK_OPTIONS`
  .. versionadded:: 3.13

  List of link options for linking the target, if it is an executable,
  shared library, or module library.  The options are passed to the
  linker as flags, in the order of appearance.

  Link options are automatically escaped for the shell.

:prop_tgt:`LINK_DEPENDS`
  List of files on which linking the target depends, if it is an executable,
  shared library, or module library.  For example, linker scripts specified
  via :prop_tgt:`LINK_OPTIONS` may be listed here such that changing them
  causes binaries to be linked again.

.. _`Target Usage Requirements`:

Target Usage Requirements
-------------------------

The *usage requirements* of a target are settings that propagate to consumers,
which link to the target via :command:`target_link_libraries`, in order to
correctly compile and link with it.  They are represented by transitive
`compile <Transitive Compile Properties_>`_ and
`link <Transitive Link Properties_>`_ properties.

Note that usage requirements are not designed as a way to make downstreams
use particular :prop_tgt:`COMPILE_OPTIONS`, :prop_tgt:`COMPILE_DEFINITIONS`,
etc. for convenience only.  The contents of the properties must be
**requirements**, not merely recommendations.

See the :ref:`Creating Relocatable Packages` section of the
:manual:`cmake-packages(7)` manual for discussion of additional care
that must be taken when specifying usage requirements while creating
packages for redistribution.

The usage requirements of a target can transitively propagate to the dependents.
The :command:`target_link_libraries` command has ``PRIVATE``,
``INTERFACE`` and ``PUBLIC`` keywords to control the propagation.

.. code-block:: cmake

  add_library(archive archive.cpp)
  target_compile_definitions(archive INTERFACE USING_ARCHIVE_LIB)

  add_library(serialization serialization.cpp)
  target_compile_definitions(serialization INTERFACE USING_SERIALIZATION_LIB)

  add_library(archiveExtras extras.cpp)
  target_link_libraries(archiveExtras PUBLIC archive)
  target_link_libraries(archiveExtras PRIVATE serialization)
  # archiveExtras is compiled with -DUSING_ARCHIVE_LIB
  # and -DUSING_SERIALIZATION_LIB

  add_executable(consumer consumer.cpp)
  # consumer is compiled with -DUSING_ARCHIVE_LIB
  target_link_libraries(consumer archiveExtras)

Because the ``archive`` is a ``PUBLIC`` dependency of ``archiveExtras``, the
usage requirements of it are propagated to ``consumer`` too.

Because
``serialization`` is a ``PRIVATE`` dependency of ``archiveExtras``, the usage
requirements of it are not propagated to ``consumer``.

Generally, a dependency should be specified in a use of
:command:`target_link_libraries` with the ``PRIVATE`` keyword if it is used by
only the implementation of a library, and not in the header files.  If a
dependency is additionally used in the header files of a library (e.g. for
class inheritance), then it should be specified as a ``PUBLIC`` dependency.
A dependency which is not used by the implementation of a library, but only by
its headers should be specified as an ``INTERFACE`` dependency.  The
:command:`target_link_libraries` command may be invoked with multiple uses of
each keyword:

.. code-block:: cmake

  target_link_libraries(archiveExtras
    PUBLIC archive
    PRIVATE serialization
  )

Usage requirements are propagated by reading the ``INTERFACE_`` variants
of target properties from dependencies and appending the values to the
non-``INTERFACE_`` variants of the operand.  For example, the
:prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` of dependencies is read and
appended to the :prop_tgt:`INCLUDE_DIRECTORIES` of the operand.  In cases
where order is relevant and maintained, and the order resulting from the
:command:`target_link_libraries` calls does not allow correct compilation,
use of an appropriate command to set the property directly may update the
order.

For example, if the linked libraries for a target must be specified
in the order ``lib1`` ``lib2`` ``lib3`` , but the include directories must
be specified in the order ``lib3`` ``lib1`` ``lib2``:

.. code-block:: cmake

  target_link_libraries(myExe lib1 lib2 lib3)
  target_include_directories(myExe
    PRIVATE $<TARGET_PROPERTY:lib3,INTERFACE_INCLUDE_DIRECTORIES>)

Note that care must be taken when specifying usage requirements for targets
which will be exported for installation using the :command:`install(EXPORT)`
command.  See :ref:`Creating Packages` for more.

.. _`Transitive Compile Properties`:

Transitive Compile Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These represent `usage requirements <Target Usage Requirements_>`_ for
compiling consumers.

:prop_tgt:`INTERFACE_COMPILE_DEFINITIONS`
  List of compile definitions for compiling sources in the target's consumers.
  Typically these are used by the target's header files.

:prop_tgt:`INTERFACE_COMPILE_OPTIONS`
  List of compile options for compiling sources in the target's consumers.

:prop_tgt:`INTERFACE_COMPILE_FEATURES`
  .. versionadded:: 3.1

  List of :manual:`compile features <cmake-compile-features(7)>` needed
  for compiling sources in the target's consumers.  Typically these
  ensure the target's header files are processed when compiling consumers
  using a sufficient language standard level.

:prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES`
  List of include directories for compiling sources in the target's consumers.
  Typically these are the locations of the target's header files.

:prop_tgt:`INTERFACE_SYSTEM_INCLUDE_DIRECTORIES`
  List of directories that, when specified as include directories, e.g., by
  :prop_tgt:`INCLUDE_DIRECTORIES` or :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES`,
  should be treated as "system" include directories when compiling sources
  in the target's consumers.

:prop_tgt:`INTERFACE_SOURCES`
  List of source files to associate with the target's consumers.

:prop_tgt:`INTERFACE_PRECOMPILE_HEADERS`
  .. versionadded:: 3.16

  List of header files to precompile and include when compiling
  sources in the target's consumers.

:prop_tgt:`INTERFACE_AUTOMOC_MACRO_NAMES`
  .. versionadded:: 3.27

  List of macro names used by :prop_tgt:`AUTOMOC` to determine if a
  C++ source in the target's consumers needs to be processed by ``moc``.

:prop_tgt:`INTERFACE_AUTOUIC_OPTIONS`
  .. versionadded:: 3.0

  List of options used by :prop_tgt:`AUTOUIC` when invoking ``uic``
  for the target's consumers.

.. _`Transitive Link Properties`:

Transitive Link Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^

These represent `usage requirements <Target Usage Requirements_>`_ for
linking consumers.

:prop_tgt:`INTERFACE_LINK_LIBRARIES`
  List of link libraries for linking the target's consumers, for
  those that are executables, shared libraries, or module libraries.
  These are the transitive dependencies of the target.

  Additionally, for compiling and linking the target's consumers,
  `usage requirements <Target Usage Requirements_>`_ are collected from
  the transitive closure of ``INTERFACE_LINK_LIBRARIES`` entries naming
  `Static Libraries`_, `Shared Libraries`_, `Interface Libraries`_,
  `Object Libraries`_, and `Imported Targets`_,

:prop_tgt:`INTERFACE_LINK_DIRECTORIES`
  .. versionadded:: 3.13

  List of link directories for linking the target's consumers, for
  those that are executables, shared libraries, or module libraries.

:prop_tgt:`INTERFACE_LINK_OPTIONS`
  .. versionadded:: 3.13

  List of link options for linking the target's consumers, for
  those that are executables, shared libraries, or module libraries.

:prop_tgt:`INTERFACE_LINK_DEPENDS`
  .. versionadded:: 3.13

  List of files on which linking the target's consumers depends, for
  those that are executables, shared libraries, or module libraries.

.. _`Custom Transitive Properties`:

Custom Transitive Properties
----------------------------

.. versionadded:: 3.30

The :genex:`TARGET_PROPERTY` generator expression evaluates the above
`build specification <Target Build Specification_>`_ and
`usage requirement <Target Usage Requirements_>`_ properties
as builtin transitive properties.  It also supports custom transitive
properties defined by the :prop_tgt:`TRANSITIVE_COMPILE_PROPERTIES`
and :prop_tgt:`TRANSITIVE_LINK_PROPERTIES` properties on the target
and its link dependencies.

For example:

.. code-block:: cmake

  add_library(example INTERFACE)
  set_target_properties(example PROPERTIES
    TRANSITIVE_COMPILE_PROPERTIES "CUSTOM_C"
    TRANSITIVE_LINK_PROPERTIES    "CUSTOM_L"

    INTERFACE_CUSTOM_C "EXAMPLE_CUSTOM_C"
    INTERFACE_CUSTOM_L "EXAMPLE_CUSTOM_L"
    )

  add_library(mylib STATIC mylib.c)
  target_link_libraries(mylib PRIVATE example)
  set_target_properties(mylib PROPERTIES
    CUSTOM_C           "MYLIB_PRIVATE_CUSTOM_C"
    CUSTOM_L           "MYLIB_PRIVATE_CUSTOM_L"
    INTERFACE_CUSTOM_C "MYLIB_IFACE_CUSTOM_C"
    INTERFACE_CUSTOM_L "MYLIB_IFACE_CUSTOM_L"
    )

  add_executable(myexe myexe.c)
  target_link_libraries(myexe PRIVATE mylib)
  set_target_properties(myexe PROPERTIES
    CUSTOM_C "MYEXE_CUSTOM_C"
    CUSTOM_L "MYEXE_CUSTOM_L"
    )

  add_custom_target(print ALL VERBATIM
    COMMAND ${CMAKE_COMMAND} -E echo
      # Prints "MYLIB_PRIVATE_CUSTOM_C;EXAMPLE_CUSTOM_C"
      "$<TARGET_PROPERTY:mylib,CUSTOM_C>"

      # Prints "MYLIB_PRIVATE_CUSTOM_L;EXAMPLE_CUSTOM_L"
      "$<TARGET_PROPERTY:mylib,CUSTOM_L>"

      # Prints "MYEXE_CUSTOM_C"
      "$<TARGET_PROPERTY:myexe,CUSTOM_C>"

      # Prints "MYEXE_CUSTOM_L;MYLIB_IFACE_CUSTOM_L;EXAMPLE_CUSTOM_L"
      "$<TARGET_PROPERTY:myexe,CUSTOM_L>"
    )

.. _`Compatible Interface Properties`:

Compatible Interface Properties
-------------------------------

Some target properties are required to be compatible between a target and
the interface of each dependency.  For example, the
:prop_tgt:`POSITION_INDEPENDENT_CODE` target property may specify a
boolean value of whether a target should be compiled as
position-independent-code, which has platform-specific consequences.
A target may also specify the usage requirement
:prop_tgt:`INTERFACE_POSITION_INDEPENDENT_CODE` to communicate that
consumers must be compiled as position-independent-code.

.. code-block:: cmake

  add_executable(exe1 exe1.cpp)
  set_property(TARGET exe1 PROPERTY POSITION_INDEPENDENT_CODE ON)

  add_library(lib1 SHARED lib1.cpp)
  set_property(TARGET lib1 PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)

  add_executable(exe2 exe2.cpp)
  target_link_libraries(exe2 lib1)

Here, both ``exe1`` and ``exe2`` will be compiled as position-independent-code.
``lib1`` will also be compiled as position-independent-code because that is the
default setting for ``SHARED`` libraries.  If dependencies have conflicting,
non-compatible requirements :manual:`cmake(1)` issues a diagnostic:

.. code-block:: cmake

  add_library(lib1 SHARED lib1.cpp)
  set_property(TARGET lib1 PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)

  add_library(lib2 SHARED lib2.cpp)
  set_property(TARGET lib2 PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE OFF)

  add_executable(exe1 exe1.cpp)
  target_link_libraries(exe1 lib1)
  set_property(TARGET exe1 PROPERTY POSITION_INDEPENDENT_CODE OFF)

  add_executable(exe2 exe2.cpp)
  target_link_libraries(exe2 lib1 lib2)

The ``lib1`` requirement ``INTERFACE_POSITION_INDEPENDENT_CODE`` is not
"compatible" with the :prop_tgt:`POSITION_INDEPENDENT_CODE` property of
the ``exe1`` target.  The library requires that consumers are built as
position-independent-code, while the executable specifies to not built as
position-independent-code, so a diagnostic is issued.

The ``lib1`` and ``lib2`` requirements are not "compatible".  One of them
requires that consumers are built as position-independent-code, while
the other requires that consumers are not built as position-independent-code.
Because ``exe2`` links to both and they are in conflict, a CMake error message
is issued::

  CMake Error: The INTERFACE_POSITION_INDEPENDENT_CODE property of "lib2" does
  not agree with the value of POSITION_INDEPENDENT_CODE already determined
  for "exe2".

To be "compatible", the :prop_tgt:`POSITION_INDEPENDENT_CODE` property,
if set must be either the same, in a boolean sense, as the
:prop_tgt:`INTERFACE_POSITION_INDEPENDENT_CODE` property of all transitively
specified dependencies on which that property is set.

This property of "compatible interface requirement" may be extended to other
properties by specifying the property in the content of the
:prop_tgt:`COMPATIBLE_INTERFACE_BOOL` target property.  Each specified property
must be compatible between the consuming target and the corresponding property
with an ``INTERFACE_`` prefix from each dependency:

.. code-block:: cmake

  add_library(lib1Version2 SHARED lib1_v2.cpp)
  set_property(TARGET lib1Version2 PROPERTY INTERFACE_CUSTOM_PROP ON)
  set_property(TARGET lib1Version2 APPEND PROPERTY
    COMPATIBLE_INTERFACE_BOOL CUSTOM_PROP
  )

  add_library(lib1Version3 SHARED lib1_v3.cpp)
  set_property(TARGET lib1Version3 PROPERTY INTERFACE_CUSTOM_PROP OFF)

  add_executable(exe1 exe1.cpp)
  target_link_libraries(exe1 lib1Version2) # CUSTOM_PROP will be ON

  add_executable(exe2 exe2.cpp)
  target_link_libraries(exe2 lib1Version2 lib1Version3) # Diagnostic

Non-boolean properties may also participate in "compatible interface"
computations.  Properties specified in the
:prop_tgt:`COMPATIBLE_INTERFACE_STRING`
property must be either unspecified or compare to the same string among
all transitively specified dependencies. This can be useful to ensure
that multiple incompatible versions of a library are not linked together
through transitive requirements of a target:

.. code-block:: cmake

  add_library(lib1Version2 SHARED lib1_v2.cpp)
  set_property(TARGET lib1Version2 PROPERTY INTERFACE_LIB_VERSION 2)
  set_property(TARGET lib1Version2 APPEND PROPERTY
    COMPATIBLE_INTERFACE_STRING LIB_VERSION
  )

  add_library(lib1Version3 SHARED lib1_v3.cpp)
  set_property(TARGET lib1Version3 PROPERTY INTERFACE_LIB_VERSION 3)

  add_executable(exe1 exe1.cpp)
  target_link_libraries(exe1 lib1Version2) # LIB_VERSION will be "2"

  add_executable(exe2 exe2.cpp)
  target_link_libraries(exe2 lib1Version2 lib1Version3) # Diagnostic

The :prop_tgt:`COMPATIBLE_INTERFACE_NUMBER_MAX` target property specifies
that content will be evaluated numerically and the maximum number among all
specified will be calculated:

.. code-block:: cmake

  add_library(lib1Version2 SHARED lib1_v2.cpp)
  set_property(TARGET lib1Version2 PROPERTY INTERFACE_CONTAINER_SIZE_REQUIRED 200)
  set_property(TARGET lib1Version2 APPEND PROPERTY
    COMPATIBLE_INTERFACE_NUMBER_MAX CONTAINER_SIZE_REQUIRED
  )

  add_library(lib1Version3 SHARED lib1_v3.cpp)
  set_property(TARGET lib1Version3 PROPERTY INTERFACE_CONTAINER_SIZE_REQUIRED 1000)

  add_executable(exe1 exe1.cpp)
  # CONTAINER_SIZE_REQUIRED will be "200"
  target_link_libraries(exe1 lib1Version2)

  add_executable(exe2 exe2.cpp)
  # CONTAINER_SIZE_REQUIRED will be "1000"
  target_link_libraries(exe2 lib1Version2 lib1Version3)

Similarly, the :prop_tgt:`COMPATIBLE_INTERFACE_NUMBER_MIN` may be used to
calculate the numeric minimum value for a property from dependencies.

Each calculated "compatible" property value may be read in the consumer at
generate-time using generator expressions.

Note that for each dependee, the set of properties specified in each
compatible interface property must not intersect with the set specified in
any of the other properties.

Property Origin Debugging
-------------------------

Because build specifications can be determined by dependencies, the lack of
locality of code which creates a target and code which is responsible for
setting build specifications may make the code more difficult to reason about.
:manual:`cmake(1)` provides a debugging facility to print the origin of the
contents of properties which may be determined by dependencies.  The properties
which can be debugged are listed in the
:variable:`CMAKE_DEBUG_TARGET_PROPERTIES` variable documentation:

.. code-block:: cmake

  set(CMAKE_DEBUG_TARGET_PROPERTIES
    INCLUDE_DIRECTORIES
    COMPILE_DEFINITIONS
    POSITION_INDEPENDENT_CODE
    CONTAINER_SIZE_REQUIRED
    LIB_VERSION
  )
  add_executable(exe1 exe1.cpp)

In the case of properties listed in :prop_tgt:`COMPATIBLE_INTERFACE_BOOL` or
:prop_tgt:`COMPATIBLE_INTERFACE_STRING`, the debug output shows which target
was responsible for setting the property, and which other dependencies also
defined the property.  In the case of
:prop_tgt:`COMPATIBLE_INTERFACE_NUMBER_MAX` and
:prop_tgt:`COMPATIBLE_INTERFACE_NUMBER_MIN`, the debug output shows the
value of the property from each dependency, and whether the value determines
the new extreme.

Build Specification with Generator Expressions
----------------------------------------------

Build specifications may use
:manual:`generator expressions <cmake-generator-expressions(7)>` containing
content which may be conditional or known only at generate-time.  For example,
the calculated "compatible" value of a property may be read with the
``TARGET_PROPERTY`` expression:

.. code-block:: cmake

  add_library(lib1Version2 SHARED lib1_v2.cpp)
  set_property(TARGET lib1Version2 PROPERTY
    INTERFACE_CONTAINER_SIZE_REQUIRED 200)
  set_property(TARGET lib1Version2 APPEND PROPERTY
    COMPATIBLE_INTERFACE_NUMBER_MAX CONTAINER_SIZE_REQUIRED
  )

  add_executable(exe1 exe1.cpp)
  target_link_libraries(exe1 lib1Version2)
  target_compile_definitions(exe1 PRIVATE
      CONTAINER_SIZE=$<TARGET_PROPERTY:CONTAINER_SIZE_REQUIRED>
  )

In this case, the ``exe1`` source files will be compiled with
``-DCONTAINER_SIZE=200``.

The unary ``TARGET_PROPERTY`` generator expression and the ``TARGET_POLICY``
generator expression are evaluated with the consuming target context.  This
means that a usage requirement specification may be evaluated differently based
on the consumer:

.. code-block:: cmake

  add_library(lib1 lib1.cpp)
  target_compile_definitions(lib1 INTERFACE
    $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:LIB1_WITH_EXE>
    $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:LIB1_WITH_SHARED_LIB>
    $<$<TARGET_POLICY:CMP0182>:CONSUMER_CMP0182_NEW>
  )

  add_executable(exe1 exe1.cpp)
  target_link_libraries(exe1 lib1)

  cmake_policy(SET CMP0182 NEW)

  add_library(shared_lib shared_lib.cpp)
  target_link_libraries(shared_lib lib1)

The ``exe1`` executable will be compiled with ``-DLIB1_WITH_EXE``, while the
``shared_lib`` shared library will be compiled with ``-DLIB1_WITH_SHARED_LIB``
and ``-DCONSUMER_CMP0182_NEW``, because policy :policy:`CMP0182` is
``NEW`` at the point where the ``shared_lib`` target is created.

The ``BUILD_INTERFACE`` expression wraps requirements which are only used when
consumed from a target in the same buildsystem, or when consumed from a target
exported to the build directory using the :command:`export` command.  The
``INSTALL_INTERFACE`` expression wraps requirements which are only used when
consumed from a target which has been installed and exported with the
:command:`install(EXPORT)` command:

.. code-block:: cmake

  add_library(ClimbingStats climbingstats.cpp)
  target_compile_definitions(ClimbingStats INTERFACE
    $<BUILD_INTERFACE:ClimbingStats_FROM_BUILD_LOCATION>
    $<INSTALL_INTERFACE:ClimbingStats_FROM_INSTALLED_LOCATION>
  )
  install(TARGETS ClimbingStats EXPORT libExport ${InstallArgs})
  install(EXPORT libExport NAMESPACE Upstream::
          DESTINATION lib/cmake/ClimbingStats)
  export(EXPORT libExport NAMESPACE Upstream::)

  add_executable(exe1 exe1.cpp)
  target_link_libraries(exe1 ClimbingStats)

In this case, the ``exe1`` executable will be compiled with
``-DClimbingStats_FROM_BUILD_LOCATION``.  The exporting commands generate
:prop_tgt:`IMPORTED` targets with either the ``INSTALL_INTERFACE`` or the
``BUILD_INTERFACE`` omitted, and the ``*_INTERFACE`` marker stripped away.
A separate project consuming the ``ClimbingStats`` package would contain:

.. code-block:: cmake

  find_package(ClimbingStats REQUIRED)

  add_executable(Downstream main.cpp)
  target_link_libraries(Downstream Upstream::ClimbingStats)

Depending on whether the ``ClimbingStats`` package was used from the build
location or the install location, the ``Downstream`` target would be compiled
with either ``-DClimbingStats_FROM_BUILD_LOCATION`` or
``-DClimbingStats_FROM_INSTALL_LOCATION``.  For more about packages and
exporting see the :manual:`cmake-packages(7)` manual.

.. _`Include Directories and Usage Requirements`:

Include Directories and Usage Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Include directories require some special consideration when specified as usage
requirements and when used with generator expressions.  The
:command:`target_include_directories` command accepts both relative and
absolute include directories:

.. code-block:: cmake

  add_library(lib1 lib1.cpp)
  target_include_directories(lib1 PRIVATE
    /absolute/path
    relative/path
  )

Relative paths are interpreted relative to the source directory where the
command appears.  Relative paths are not allowed in the
:prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` of :prop_tgt:`IMPORTED` targets.

In cases where a non-trivial generator expression is used, the
``INSTALL_PREFIX`` expression may be used within the argument of an
``INSTALL_INTERFACE`` expression.  It is a replacement marker which
expands to the installation prefix when imported by a consuming project.

Include directories usage requirements commonly differ between the build-tree
and the install-tree.  The ``BUILD_INTERFACE`` and ``INSTALL_INTERFACE``
generator expressions can be used to describe separate usage requirements
based on the usage location.  Relative paths are allowed within the
``INSTALL_INTERFACE`` expression and are interpreted relative to the
installation prefix.  For example:

.. code-block:: cmake

  add_library(ClimbingStats climbingstats.cpp)
  target_include_directories(ClimbingStats INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:/absolute/path>
    $<INSTALL_INTERFACE:relative/path>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/$<CONFIG>/generated>
  )

Two convenience APIs are provided relating to include directories usage
requirements.  The :variable:`CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE` variable
may be enabled, with an equivalent effect to:

.. code-block:: cmake

  set_property(TARGET tgt APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_BINARY_DIR}>
  )

for each target affected.  The convenience for installed targets is
an ``INCLUDES DESTINATION`` component with the :command:`install(TARGETS)`
command:

.. code-block:: cmake

  install(TARGETS foo bar bat EXPORT tgts ${dest_args}
    INCLUDES DESTINATION include
  )
  install(EXPORT tgts ${other_args})
  install(FILES ${headers} DESTINATION include)

This is equivalent to appending ``${CMAKE_INSTALL_PREFIX}/include`` to the
:prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` of each of the installed
:prop_tgt:`IMPORTED` targets when generated by :command:`install(EXPORT)`.

When the :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` of an
:ref:`imported target <Imported targets>` is consumed, the entries in the
property may be treated as system include directories.  The effects of that
are toolchain-dependent, but one common effect is to omit compiler warnings
for headers found in those directories.  The :prop_tgt:`SYSTEM` property of
the installed target determines this behavior (see the
:prop_tgt:`EXPORT_NO_SYSTEM` property for how to modify the installed value
for a target).  It is also possible to change how consumers interpret the
system behavior of consumed imported targets by setting the
:prop_tgt:`NO_SYSTEM_FROM_IMPORTED` target property on the *consumer*.

If a binary target is linked transitively to a macOS :prop_tgt:`FRAMEWORK`, the
``Headers`` directory of the framework is also treated as a usage requirement.
This has the same effect as passing the framework directory as an include
directory.

Link Libraries and Generator Expressions
----------------------------------------

Like build specifications, :prop_tgt:`link libraries <LINK_LIBRARIES>` may be
specified with generator expression conditions.  However, as consumption of
usage requirements is based on collection from linked dependencies, there is
an additional limitation that the link dependencies must form a "directed
acyclic graph".  That is, if linking to a target is dependent on the value of
a target property, that target property may not be dependent on the linked
dependencies:

.. code-block:: cmake

  add_library(lib1 lib1.cpp)
  add_library(lib2 lib2.cpp)
  target_link_libraries(lib1 PUBLIC
    $<$<TARGET_PROPERTY:POSITION_INDEPENDENT_CODE>:lib2>
  )
  add_library(lib3 lib3.cpp)
  set_property(TARGET lib3 PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)

  add_executable(exe1 exe1.cpp)
  target_link_libraries(exe1 lib1 lib3)

As the value of the :prop_tgt:`POSITION_INDEPENDENT_CODE` property of
the ``exe1`` target is dependent on the linked libraries (``lib3``), and the
edge of linking ``exe1`` is determined by the same
:prop_tgt:`POSITION_INDEPENDENT_CODE` property, the dependency graph above
contains a cycle.  :manual:`cmake(1)` issues an error message.

.. _`Output Artifacts`:

Output Artifacts
----------------

The buildsystem targets created by the :command:`add_library` and
:command:`add_executable` commands create rules to create binary outputs.
The exact output location of the binaries can only be determined at
generate-time because it can depend on the build-configuration and the
link-language of linked dependencies etc.  ``TARGET_FILE``,
``TARGET_LINKER_FILE`` and related expressions can be used to access the
name and location of generated binaries.  These expressions do not work
for ``OBJECT`` libraries however, as there is no single file generated
by such libraries which is relevant to the expressions.

There are three kinds of output artifacts that may be build by targets
as detailed in the following sections.  Their classification differs
between DLL platforms and non-DLL platforms.  All Windows-based
systems including Cygwin are DLL platforms.

.. _`Runtime Output Artifacts`:

Runtime Output Artifacts
^^^^^^^^^^^^^^^^^^^^^^^^

A *runtime* output artifact of a buildsystem target may be:

* The executable file (e.g. ``.exe``) of an executable target
  created by the :command:`add_executable` command.

* On DLL platforms: the executable file (e.g. ``.dll``) of a shared
  library target created by the :command:`add_library` command
  with the ``SHARED`` option.

The :prop_tgt:`RUNTIME_OUTPUT_DIRECTORY` and :prop_tgt:`RUNTIME_OUTPUT_NAME`
target properties may be used to control runtime output artifact locations
and names in the build tree.

.. _`Library Output Artifacts`:

Library Output Artifacts
^^^^^^^^^^^^^^^^^^^^^^^^

A *library* output artifact of a buildsystem target may be:

* The loadable module file (e.g. ``.dll`` or ``.so``) of a module
  library target created by the :command:`add_library` command
  with the ``MODULE`` option.

* On non-DLL platforms: the shared library file (e.g. ``.so`` or ``.dylib``)
  of a shared library target created by the :command:`add_library`
  command with the ``SHARED`` option.

The :prop_tgt:`LIBRARY_OUTPUT_DIRECTORY` and :prop_tgt:`LIBRARY_OUTPUT_NAME`
target properties may be used to control library output artifact locations
and names in the build tree.

.. _`Archive Output Artifacts`:

Archive Output Artifacts
^^^^^^^^^^^^^^^^^^^^^^^^

An *archive* output artifact of a buildsystem target may be:

* The static library file (e.g. ``.lib`` or ``.a``) of a static
  library target created by the :command:`add_library` command
  with the ``STATIC`` option.

* On DLL platforms: the import library file (e.g. ``.lib``) of a shared
  library target created by the :command:`add_library` command
  with the ``SHARED`` option.  This file is only guaranteed to exist if
  the library exports at least one unmanaged symbol.

* On DLL platforms: the import library file (e.g. ``.lib``) of an
  executable target created by the :command:`add_executable` command
  when its :prop_tgt:`ENABLE_EXPORTS` target property is set.

* On AIX: the linker import file (e.g. ``.imp``) of an executable target
  created by the :command:`add_executable` command when its
  :prop_tgt:`ENABLE_EXPORTS` target property is set.

* On macOS: the linker import file (e.g. ``.tbd``) of a shared library target
  created by the :command:`add_library` command with the ``SHARED`` option and
  when its :prop_tgt:`ENABLE_EXPORTS` target property is set.

The :prop_tgt:`ARCHIVE_OUTPUT_DIRECTORY` and :prop_tgt:`ARCHIVE_OUTPUT_NAME`
target properties may be used to control archive output artifact locations
and names in the build tree.

Directory-Scoped Commands
-------------------------

The :command:`target_include_directories`,
:command:`target_compile_definitions` and
:command:`target_compile_options` commands have an effect on only one
target at a time.  The commands :command:`add_compile_definitions`,
:command:`add_compile_options` and :command:`include_directories` have
a similar function, but operate at directory scope instead of target
scope for convenience.

.. _`Build Configurations`:

Build Configurations
====================

Configurations determine specifications for a certain type of build, such
as ``Release`` or ``Debug``.  The way this is specified depends on the type
of :manual:`generator <cmake-generators(7)>` being used.  For single
configuration generators like  :ref:`Makefile Generators` and
:generator:`Ninja`, the configuration is specified at configure time by the
:variable:`CMAKE_BUILD_TYPE` variable. For multi-configuration generators
like :ref:`Visual Studio <Visual Studio Generators>`, :generator:`Xcode`, and
:generator:`Ninja Multi-Config`, the configuration is chosen by the user at
build time and :variable:`CMAKE_BUILD_TYPE` is ignored.  In the
multi-configuration case, the set of *available* configurations is specified
at configure time by the :variable:`CMAKE_CONFIGURATION_TYPES` variable,
but the actual configuration used cannot be known until the build stage.
This difference is often misunderstood, leading to problematic code like the
following:

.. code-block:: cmake

  # WARNING: This is wrong for multi-config generators because they don't use
  #          and typically don't even set CMAKE_BUILD_TYPE
  string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
  if (build_type STREQUAL debug)
    target_compile_definitions(exe1 PRIVATE DEBUG_BUILD)
  endif()

:manual:`Generator expressions <cmake-generator-expressions(7)>` should be
used instead to handle configuration-specific logic correctly, regardless of
the generator used.  For example:

.. code-block:: cmake

  # Works correctly for both single and multi-config generators
  target_compile_definitions(exe1 PRIVATE
    $<$<CONFIG:Debug>:DEBUG_BUILD>
  )

In the presence of :prop_tgt:`IMPORTED` targets, the content of
:prop_tgt:`MAP_IMPORTED_CONFIG_DEBUG <MAP_IMPORTED_CONFIG_<CONFIG>>` is also
accounted for by the above :genex:`$<CONFIG:Debug>` expression.


Case Sensitivity
----------------

:variable:`CMAKE_BUILD_TYPE` and :variable:`CMAKE_CONFIGURATION_TYPES` are
just like other variables in that any string comparisons made with their
values will be case-sensitive.  The :genex:`$<CONFIG>` generator expression also
preserves the casing of the configuration as set by the user or CMake defaults.
For example:

.. code-block:: cmake

  # NOTE: Don't use these patterns, they are for illustration purposes only.

  set(CMAKE_BUILD_TYPE Debug)
  if(CMAKE_BUILD_TYPE STREQUAL DEBUG)
    # ... will never get here, "Debug" != "DEBUG"
  endif()
  add_custom_target(print_config ALL
    # Prints "Config is Debug" in this single-config case
    COMMAND ${CMAKE_COMMAND} -E echo "Config is $<CONFIG>"
    VERBATIM
  )

  set(CMAKE_CONFIGURATION_TYPES Debug Release)
  if(DEBUG IN_LIST CMAKE_CONFIGURATION_TYPES)
    # ... will never get here, "Debug" != "DEBUG"
  endif()

In contrast, CMake treats the configuration type case-insensitively when
using it internally in places that modify behavior based on the configuration.
For example, the :genex:`$<CONFIG:Debug>` generator expression will evaluate to 1
for a configuration of not only ``Debug``, but also ``DEBUG``, ``debug`` or
even ``DeBuG``.  Therefore, you can specify configuration types in
:variable:`CMAKE_BUILD_TYPE` and :variable:`CMAKE_CONFIGURATION_TYPES` with
any mixture of upper and lowercase, although there are strong conventions
(see the next section).  If you must test the value in string comparisons,
always convert the value to upper or lowercase first and adjust the test
accordingly.

Default And Custom Configurations
---------------------------------

By default, CMake defines a number of standard configurations:

* ``Debug``
* ``Release``
* ``RelWithDebInfo``
* ``MinSizeRel``

In multi-config generators, the :variable:`CMAKE_CONFIGURATION_TYPES` variable
will be populated with (potentially a subset of) the above list by default,
unless overridden by the project or user.  The actual configuration used is
selected by the user at build time.

For single-config generators, the configuration is specified with the
:variable:`CMAKE_BUILD_TYPE` variable at configure time and cannot be changed
at build time.  The default value will often be none of the above standard
configurations and will instead be an empty string.  A common misunderstanding
is that this is the same as ``Debug``, but that is not the case.  Users should
always explicitly specify the build type instead to avoid this common problem.

The above standard configuration types provide reasonable behavior on most
platforms, but they can be extended to provide other types.  Each configuration
defines a set of compiler and linker flag variables for the language in use.
These variables follow the convention :variable:`CMAKE_<LANG>_FLAGS_<CONFIG>`,
where ``<CONFIG>`` is always the uppercase configuration name.  When defining
a custom configuration type, make sure these variables are set appropriately,
typically as cache variables.


Pseudo Targets
==============

Some target types do not represent outputs of the buildsystem, but only inputs
such as external dependencies, aliases or other non-build artifacts.  Pseudo
targets are not represented in the generated buildsystem.

.. _`Imported Targets`:

Imported Targets
----------------

An :prop_tgt:`IMPORTED` target represents a pre-existing dependency.  Usually
such targets are defined by an upstream package and should be treated as
immutable. After declaring an :prop_tgt:`IMPORTED` target one can adjust its
target properties by using the customary commands such as
:command:`target_compile_definitions`, :command:`target_include_directories`,
:command:`target_compile_options` or :command:`target_link_libraries` just like
with any other regular target.

:prop_tgt:`IMPORTED` targets may have the same usage requirement properties
populated as binary targets, such as
:prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES`,
:prop_tgt:`INTERFACE_COMPILE_DEFINITIONS`,
:prop_tgt:`INTERFACE_COMPILE_OPTIONS`,
:prop_tgt:`INTERFACE_LINK_LIBRARIES`, and
:prop_tgt:`INTERFACE_POSITION_INDEPENDENT_CODE`.

The :prop_tgt:`LOCATION` may also be read from an IMPORTED target, though there
is rarely reason to do so.  Commands such as :command:`add_custom_command` can
transparently use an :prop_tgt:`IMPORTED` :prop_tgt:`EXECUTABLE <TYPE>` target
as a ``COMMAND`` executable.

The scope of the definition of an :prop_tgt:`IMPORTED` target is the directory
where it was defined.  It may be accessed and used from subdirectories, but
not from parent directories or sibling directories.  The scope is similar to
the scope of a cmake variable.

It is also possible to define a ``GLOBAL`` :prop_tgt:`IMPORTED` target which is
accessible globally in the buildsystem.

See the :manual:`cmake-packages(7)` manual for more on creating packages
with :prop_tgt:`IMPORTED` targets.

.. _`Alias Targets`:

Alias Targets
-------------

An ``ALIAS`` target is a name which may be used interchangeably with
a binary target name in read-only contexts.  A primary use-case for ``ALIAS``
targets is for example or unit test executables accompanying a library, which
may be part of the same buildsystem or built separately based on user
configuration.

.. code-block:: cmake

  add_library(lib1 lib1.cpp)
  install(TARGETS lib1 EXPORT lib1Export ${dest_args})
  install(EXPORT lib1Export NAMESPACE Upstream:: ${other_args})

  add_library(Upstream::lib1 ALIAS lib1)

In another directory, we can link unconditionally to the ``Upstream::lib1``
target, which may be an :prop_tgt:`IMPORTED` target from a package, or an
``ALIAS`` target if built as part of the same buildsystem.

.. code-block:: cmake

  if (NOT TARGET Upstream::lib1)
    find_package(lib1 REQUIRED)
  endif()
  add_executable(exe1 exe1.cpp)
  target_link_libraries(exe1 Upstream::lib1)

``ALIAS`` targets are not mutable, installable or exportable.  They are
entirely local to the buildsystem description.  A name can be tested for
whether it is an ``ALIAS`` name by reading the :prop_tgt:`ALIASED_TARGET`
property from it:

.. code-block:: cmake

  get_target_property(_aliased Upstream::lib1 ALIASED_TARGET)
  if(_aliased)
    message(STATUS "The name Upstream::lib1 is an ALIAS for ${_aliased}.")
  endif()

.. _`Interface Libraries`:

Interface Libraries
-------------------

An ``INTERFACE`` library target does not compile sources and does not
produce a library artifact on disk, so it has no :prop_tgt:`LOCATION`.

It may specify `usage requirements <Target Usage Requirements_>`_,
`compatible interface properties <Compatible Interface Properties_>`_, and
`custom transitive properties <Custom Transitive Properties_>`_.
Only the ``INTERFACE`` modes of the :command:`target_include_directories`,
:command:`target_compile_definitions`, :command:`target_compile_options`,
:command:`target_sources`, and :command:`target_link_libraries` commands
may be used with ``INTERFACE`` libraries.

Since CMake 3.19, an ``INTERFACE`` library target may optionally contain
source files.  An interface library that contains source files will be
included as a build target in the generated buildsystem.  It does not
compile sources, but may contain custom commands to generate other sources.
Additionally, IDEs will show the source files as part of the target for
interactive reading and editing.

A primary use-case for ``INTERFACE`` libraries is header-only libraries.
Since CMake 3.23, header files may be associated with a library by adding
them to a header set using the :command:`target_sources` command:

.. code-block:: cmake

  add_library(Eigen INTERFACE)

  target_sources(Eigen PUBLIC
    FILE_SET HEADERS
      BASE_DIRS src
      FILES src/eigen.h src/vector.h src/matrix.h
  )

  add_executable(exe1 exe1.cpp)
  target_link_libraries(exe1 Eigen)

When we specify the ``FILE_SET`` here, the ``BASE_DIRS`` we define automatically
become include directories in the usage requirements for the target ``Eigen``.
The usage requirements from the target are consumed and used when compiling, but
have no effect on linking.

Another use-case is to employ an entirely target-focussed design for usage
requirements:

.. code-block:: cmake

  add_library(pic_on INTERFACE)
  set_property(TARGET pic_on PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
  add_library(pic_off INTERFACE)
  set_property(TARGET pic_off PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE OFF)

  add_library(enable_rtti INTERFACE)
  target_compile_options(enable_rtti INTERFACE
    $<$<OR:$<COMPILER_ID:GNU>,$<COMPILER_ID:Clang>>:-rtti>
  )

  add_executable(exe1 exe1.cpp)
  target_link_libraries(exe1 pic_on enable_rtti)

This way, the build specification of ``exe1`` is expressed entirely as linked
targets, and the complexity of compiler-specific flags is encapsulated in an
``INTERFACE`` library target.

``INTERFACE`` libraries may be installed and exported. We can install the
default header set along with the target:

.. code-block:: cmake

  add_library(Eigen INTERFACE)

  target_sources(Eigen PUBLIC
    FILE_SET HEADERS
      BASE_DIRS src
      FILES src/eigen.h src/vector.h src/matrix.h
  )

  install(TARGETS Eigen EXPORT eigenExport
    FILE_SET HEADERS DESTINATION include/Eigen)
  install(EXPORT eigenExport NAMESPACE Upstream::
    DESTINATION lib/cmake/Eigen
  )

Here, the headers defined in the header set are installed to ``include/Eigen``.
The install destination automatically becomes an include directory that is a
usage requirement for consumers.

Properties Allowed on Interface Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since CMake 3.19, interface libraries allow setting or reading target
properties with any name, just like other target kinds always have.

Prior to CMake 3.19, interface libraries only allowed setting or reading
target properties with a limited set of names:

* Properties named with an ``INTERFACE_`` prefix, either builtin
  `usage requirements <Target Usage Requirements_>`_, or custom names.

* Built-in properties named with a ``COMPATIBLE_INTERFACE_`` prefix
  (`compatible interface properties <Compatible Interface Properties_>`_).

* Built-in properties :prop_tgt:`NAME`, :prop_tgt:`EXPORT_NAME`,
  :prop_tgt:`EXPORT_PROPERTIES`, :prop_tgt:`MANUALLY_ADDED_DEPENDENCIES`,
  :prop_tgt:`IMPORTED`, :prop_tgt:`IMPORTED_LIBNAME_<CONFIG>`, and
  :prop_tgt:`MAP_IMPORTED_CONFIG_<CONFIG>`.

* .. versionadded:: 3.11
    Properties named with a leading underscore (``_``)
    or lowercase ASCII character.
