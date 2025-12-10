target_link_libraries
---------------------

.. only:: html

   .. contents::

Specify libraries or flags to use when linking a given target and/or
its dependents.  :ref:`Usage requirements <Target Usage Requirements>`
from linked library targets will be propagated.  Usage requirements
of a target's dependencies affect compilation of its own sources.

Overview
^^^^^^^^

This command has several signatures as detailed in subsections below.
All of them have the general form

.. code-block:: cmake

  target_link_libraries(<target> ... <item>... ...)

The named ``<target>`` must have been created by a command such as
:command:`add_executable` or :command:`add_library` and must not be an
:ref:`ALIAS target <Alias Targets>`.  If policy :policy:`CMP0079` is not
set to ``NEW`` then the target must have been created in the current
directory.  Repeated calls for the same ``<target>`` append items in
the order called.

.. versionadded:: 3.13
  The ``<target>`` doesn't have to be defined in the same directory as the
  ``target_link_libraries`` call.

Each ``<item>`` may be:

* **A library target name**: The generated link line will have the
  full path to the linkable library file associated with the target.
  The buildsystem will have a dependency to re-link ``<target>`` if
  the library file changes.

  The named target must be created by :command:`add_library` within
  the project or as an :ref:`IMPORTED library <Imported Targets>`.
  If it is created within the project an ordering dependency will
  automatically be added in the build system to make sure the named
  library target is up-to-date before the ``<target>`` links.

  If an imported library has the :prop_tgt:`IMPORTED_NO_SONAME`
  target property set, CMake may ask the linker to search for
  the library instead of using the full path
  (e.g. ``/usr/lib/libfoo.so`` becomes ``-lfoo``).

  The full path to the target's artifact will be quoted/escaped for
  the shell automatically.

* **A full path to a library file**: The generated link line will
  normally preserve the full path to the file. The buildsystem will
  have a dependency to re-link ``<target>`` if the library file changes.

  There are some cases where CMake may ask the linker to search for
  the library (e.g. ``/usr/lib/libfoo.so`` becomes ``-lfoo``), such
  as when a shared library is detected to have no ``SONAME`` field.
  In CMake versions prior to 4.0, see policy :policy:`CMP0060` for
  discussion of another case.

  If the library file is in a macOS framework, the ``Headers`` directory
  of the framework will also be processed as a
  :ref:`usage requirement <Target Usage Requirements>`.  This has the same
  effect as passing the framework directory as an include directory.

  .. versionadded:: 3.28

    The library file may point to a ``.xcframework`` folder on Apple platforms.
    If it does, the target will get the selected library's ``Headers``
    directory as a usage requirement.

  .. versionadded:: 3.8
    On :ref:`Visual Studio Generators` for VS 2010 and above, library files
    ending in ``.targets`` will be treated as MSBuild targets files and
    imported into generated project files.  This is not supported by other
    generators.

  The full path to the library file will be quoted/escaped for
  the shell automatically.

* **A plain library name**: The generated link line will ask the linker
  to search for the library (e.g. ``foo`` becomes ``-lfoo`` or ``foo.lib``).

  The library name/flag is treated as a command-line string fragment and
  will be used with no extra quoting or escaping.

* **A link flag**: Item names starting with ``-``, but not ``-l`` or
  ``-framework``, are treated as linker flags.  Note that such flags will
  be treated like any other library link item for purposes of transitive
  dependencies, so they are generally safe to specify only as private link
  items that will not propagate to dependents.

  Link flags specified here are inserted into the link command in the same
  place as the link libraries. This might not be correct, depending on
  the linker. Use the :prop_tgt:`LINK_OPTIONS` target property or
  :command:`target_link_options` command to add link
  flags explicitly. The flags will then be placed at the toolchain-defined
  flag position in the link command.

  .. versionadded:: 3.13
    :prop_tgt:`LINK_OPTIONS` target property and :command:`target_link_options`
    command.  For earlier versions of CMake, use :prop_tgt:`LINK_FLAGS`
    property instead.

  The link flag is treated as a command-line string fragment and
  will be used with no extra quoting or escaping.

* **A generator expression**: A ``$<...>`` :manual:`generator expression
  <cmake-generator-expressions(7)>` may evaluate to any of the above
  items or to a :ref:`semicolon-separated list <CMake Language Lists>` of them.
  If the ``...`` contains any ``;`` characters, e.g. after evaluation
  of a ``${list}`` variable, be sure to use an explicitly quoted
  argument ``"$<...>"`` so that this command receives it as a
  single ``<item>``.

  Additionally, a generator expression may be used as a fragment of
  any of the above items, e.g. ``foo$<1:_d>``.

* A ``debug``, ``optimized``, or ``general`` keyword immediately followed
  by another ``<item>``.  The item following such a keyword will be used
  only for the corresponding build configuration.  The ``debug`` keyword
  corresponds to the ``Debug`` configuration (or to configurations named
  in the :prop_gbl:`DEBUG_CONFIGURATIONS` global property if it is set).
  The ``optimized`` keyword corresponds to all other configurations.  The
  ``general`` keyword corresponds to all configurations, and is purely
  optional.  These keywords are interpreted immediately by this command and
  therefore have no special meaning when produced by a generator expression.

  Alternatively, generator expressions like :genex:`$<CONFIG>` provide finer
  per-configuration linking of ``<item>``.  For a more structured approach,
  higher granularity can be achieved by creating and linking to
  :ref:`IMPORTED library targets <Imported Targets>` with the
  :prop_tgt:`IMPORTED_CONFIGURATIONS` property set, particularly in find
  modules.

Items containing ``::``, such as ``Foo::Bar``, are assumed to be
:ref:`IMPORTED <Imported Targets>` or :ref:`ALIAS <Alias Targets>` library
target names and will cause an error if no such target exists.
See policy :policy:`CMP0028`.

See the :variable:`CMAKE_LINK_LIBRARIES_STRATEGY` variable and
corresponding :prop_tgt:`LINK_LIBRARIES_STRATEGY` target property
for details on how CMake orders direct link dependencies on linker
command lines.

See the :manual:`cmake-buildsystem(7)` manual for more on defining
buildsystem properties.

.. include:: ../command/include/LINK_LIBRARIES_LINKER.rst

Libraries for a Target and/or its Dependents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

  target_link_libraries(<target>
                        <PRIVATE|PUBLIC|INTERFACE> <item>...
                       [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)

The ``PUBLIC``, ``PRIVATE`` and ``INTERFACE``
:ref:`scope <Target Command Scope>` keywords can be used to
specify both the link dependencies and the link interface in one command.

Libraries and targets following ``PUBLIC`` are linked to, and are made
part of the link interface.  Libraries and targets following ``PRIVATE``
are linked to, but are not made part of the link interface.  Libraries
following ``INTERFACE`` are appended to the link interface and are not
used for linking ``<target>``.

Libraries for both a Target and its Dependents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

  target_link_libraries(<target> <item>...)

Library dependencies are transitive by default with this signature.
When this target is linked into another target then the libraries
linked to this target will appear on the link line for the other
target too.  This transitive "link interface" is stored in the
:prop_tgt:`INTERFACE_LINK_LIBRARIES` target property and may be overridden
by setting the property directly.

In CMake versions prior to 4.0, if :policy:`CMP0022` is not set to ``NEW``,
transitive linking is built in but may be overridden by the
:prop_tgt:`LINK_INTERFACE_LIBRARIES` property.  Calls to other signatures
of this command may set the property making any libraries linked
exclusively by this signature private.

Libraries for a Target and/or its Dependents (Legacy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This signature is for compatibility only.  Prefer the ``PUBLIC`` or
``PRIVATE`` keywords instead.

.. code-block:: cmake

  target_link_libraries(<target>
                        <LINK_PRIVATE|LINK_PUBLIC> <lib>...
                       [<LINK_PRIVATE|LINK_PUBLIC> <lib>...]...)

The ``LINK_PUBLIC`` and ``LINK_PRIVATE`` modes can be used to specify both
the link dependencies and the link interface in one command.

Libraries and targets following ``LINK_PUBLIC`` are linked to, and are
made part of the :prop_tgt:`INTERFACE_LINK_LIBRARIES`.

In CMake versions prior to 4.0, if policy :policy:`CMP0022` is not ``NEW``,
they are also made part of the :prop_tgt:`LINK_INTERFACE_LIBRARIES`.
Libraries and targets following ``LINK_PRIVATE`` are linked to, but are
not made part of the :prop_tgt:`INTERFACE_LINK_LIBRARIES`
(or :prop_tgt:`LINK_INTERFACE_LIBRARIES`).

Libraries for Dependents Only (Legacy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This signature is for compatibility only.  Prefer the ``INTERFACE`` mode
instead.

.. code-block:: cmake

  target_link_libraries(<target> LINK_INTERFACE_LIBRARIES <item>...)

The ``LINK_INTERFACE_LIBRARIES`` mode appends the libraries to the
:prop_tgt:`INTERFACE_LINK_LIBRARIES` target property instead of using them
for linking.

In CMake versions prior to 4.0, if policy :policy:`CMP0022` is not ``NEW``,
then this mode also appends libraries to the
:prop_tgt:`LINK_INTERFACE_LIBRARIES` and its per-configuration equivalent.

.. _`Linking Object Libraries`:

Linking Object Libraries
^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.12

:ref:`Object Libraries` may be used as the ``<target>`` (first) argument
of ``target_link_libraries`` to specify dependencies of their sources
on other libraries.  For example, the code

.. code-block:: cmake

  add_library(A SHARED a.c)
  target_compile_definitions(A PUBLIC A)

  add_library(obj OBJECT obj.c)
  target_compile_definitions(obj PUBLIC OBJ)
  target_link_libraries(obj PUBLIC A)

compiles ``obj.c`` with ``-DA -DOBJ`` and establishes usage requirements
for ``obj`` that propagate to its dependents.

Normal libraries and executables may link to :ref:`Object Libraries`
to get their objects and usage requirements.  Continuing the above
example, the code

.. code-block:: cmake

  add_library(B SHARED b.c)
  target_link_libraries(B PUBLIC obj)

compiles ``b.c`` with ``-DA -DOBJ``, creates shared library ``B``
with object files from ``b.c`` and ``obj.c``, and links ``B`` to ``A``.
Furthermore, the code

.. code-block:: cmake

  add_executable(main main.c)
  target_link_libraries(main B)

compiles ``main.c`` with ``-DA -DOBJ`` and links executable ``main``
to ``B`` and ``A``.  The object library's usage requirements are
propagated transitively through ``B``, but its object files are not.

:ref:`Object Libraries` may "link" to other object libraries to get
usage requirements, but since they do not have a link step nothing
is done with their object files.  Continuing from the above example,
the code:

.. code-block:: cmake

  add_library(obj2 OBJECT obj2.c)
  target_link_libraries(obj2 PUBLIC obj)

  add_executable(main2 main2.c)
  target_link_libraries(main2 obj2)

compiles ``obj2.c`` with ``-DA -DOBJ``, creates executable ``main2``
with object files from ``main2.c`` and ``obj2.c``, and links ``main2``
to ``A``.

In other words, when :ref:`Object Libraries` appear in a target's
:prop_tgt:`INTERFACE_LINK_LIBRARIES` property they will be
treated as :ref:`Interface Libraries`, but when they appear in
a target's :prop_tgt:`LINK_LIBRARIES` property their object files
will be included in the link too.

.. _`Linking Object Libraries via $<TARGET_OBJECTS>`:

Linking Object Libraries via ``$<TARGET_OBJECTS>``
""""""""""""""""""""""""""""""""""""""""""""""""""

.. versionadded:: 3.21

The object files associated with an object library may be referenced
by the :genex:`$<TARGET_OBJECTS>` generator expression.  Such object
files are placed on the link line *before* all libraries, regardless
of their relative order.  Additionally, an ordering dependency will be
added to the build system to make sure the object library is up-to-date
before the dependent target links.  For example, the code

.. code-block:: cmake

  add_library(obj3 OBJECT obj3.c)
  target_compile_definitions(obj3 PUBLIC OBJ3)

  add_executable(main3 main3.c)
  target_link_libraries(main3 PRIVATE a3 $<TARGET_OBJECTS:obj3> b3)

links executable ``main3`` with object files from ``main3.c``
and ``obj3.c`` followed by the ``a3`` and ``b3`` libraries.
``main3.c`` is *not* compiled with usage requirements from ``obj3``,
such as ``-DOBJ3``.

This approach can be used to achieve transitive inclusion of object
files in link lines as usage requirements.  Continuing the above
example, the code

.. code-block:: cmake

  add_library(iface_obj3 INTERFACE)
  target_link_libraries(iface_obj3 INTERFACE obj3 $<TARGET_OBJECTS:obj3>)

creates an interface library ``iface_obj3`` that forwards the ``obj3``
usage requirements and adds the ``obj3`` object files to dependents'
link lines.  The code

.. code-block:: cmake

  add_executable(use_obj3 use_obj3.c)
  target_link_libraries(use_obj3 PRIVATE iface_obj3)

compiles ``use_obj3.c`` with ``-DOBJ3`` and links executable ``use_obj3``
with object files from ``use_obj3.c`` and ``obj3.c``.

This also works transitively through a static library.  Since a static
library does not link, it does not consume the object files from
object libraries referenced this way.  Instead, the object files
become transitive link dependencies of the static library.
Continuing the above example, the code

.. code-block:: cmake

  add_library(static3 STATIC static3.c)
  target_link_libraries(static3 PRIVATE iface_obj3)

  add_executable(use_static3 use_static3.c)
  target_link_libraries(use_static3 PRIVATE static3)

compiles ``static3.c`` with ``-DOBJ3`` and creates ``libstatic3.a``
using only its own object file.  ``use_static3.c`` is compiled *without*
``-DOBJ3`` because the usage requirement is not transitive through
the private dependency of ``static3``.  However, the link dependencies
of ``static3`` are propagated, including the ``iface_obj3`` reference
to ``$<TARGET_OBJECTS:obj3>``.  The ``use_static3`` executable is
created with object files from ``use_static3.c`` and ``obj3.c``, and
linked to library ``libstatic3.a``.

When using this approach, it is the project's responsibility to avoid
linking multiple dependent binaries to ``iface_obj3``, because they will
all get the ``obj3`` object files on their link lines.

.. note::

  Referencing :genex:`$<TARGET_OBJECTS>` in ``target_link_libraries``
  calls worked in versions of CMake prior to 3.21 for some cases,
  but was not fully supported:

  * It did not place the object files before libraries on link lines.
  * It did not add an ordering dependency on the object library.
  * It did not work in Xcode with multiple architectures.

Cyclic Dependencies of Static Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library dependency graph is normally acyclic (a DAG), but in the case
of mutually-dependent ``STATIC`` libraries CMake allows the graph to
contain cycles (strongly connected components).  When another target links
to one of the libraries, CMake repeats the entire connected component.
For example, the code

.. code-block:: cmake

  add_library(A STATIC a.c)
  add_library(B STATIC b.c)
  target_link_libraries(A B)
  target_link_libraries(B A)
  add_executable(main main.c)
  target_link_libraries(main A)

links ``main`` to ``A B A B``.  While one repetition is usually
sufficient, pathological object file and symbol arrangements can require
more.  One may handle such cases by using the
:prop_tgt:`LINK_INTERFACE_MULTIPLICITY` target property or by manually
repeating the component in the last ``target_link_libraries`` call.
However, if two archives are really so interdependent they should probably
be combined into a single archive, perhaps by using :ref:`Object Libraries`.

Creating Relocatable Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |INTERFACE_PROPERTY_LINK| replace:: :prop_tgt:`INTERFACE_LINK_LIBRARIES`
.. include:: /include/INTERFACE_LINK_LIBRARIES_WARNING.rst

See Also
^^^^^^^^

* :command:`target_compile_definitions`
* :command:`target_compile_features`
* :command:`target_compile_options`
* :command:`target_include_directories`
* :command:`target_link_directories`
* :command:`target_link_options`
* :command:`target_precompile_headers`
* :command:`target_sources`
