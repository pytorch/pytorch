add_library
-----------

.. only:: html

   .. contents::

Add a library to the project using the specified source files.

Normal Libraries
^^^^^^^^^^^^^^^^

.. signature::
  add_library(<name> [<type>] [EXCLUDE_FROM_ALL] <sources>...)
  :target: normal

  Add a library target called ``<name>`` to be built from the source files
  listed in the command invocation.

  The optional ``<type>`` specifies the type of library to be created:

  ``STATIC``
    A :ref:`Static Library <Static Libraries>`:
    an archive of object files for use when linking other targets.

  ``SHARED``
    A :ref:`Shared Library <Shared Libraries>`:
    a dynamic library that may be linked by other targets and loaded
    at runtime.

  ``MODULE``
    A :ref:`Module Library <Module Libraries>`:
    a plugin that may not be linked by other targets, but may be
    dynamically loaded at runtime using dlopen-like functionality.

  If no ``<type>`` is given the default is ``STATIC`` or ``SHARED``
  based on the value of the :variable:`BUILD_SHARED_LIBS` variable.

  The options are:

  ``EXCLUDE_FROM_ALL``
    Set the :prop_tgt:`EXCLUDE_FROM_ALL` target property automatically.
    See documentation of that target property for details.

The ``<name>`` corresponds to the logical target name and must be globally
unique within a project.  The actual file name of the library built is
constructed based on conventions of the native platform (such as
``lib<name>.a`` or ``<name>.lib``).

.. versionadded:: 3.1
  Source arguments to ``add_library`` may use "generator expressions" with
  the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
  manual for available expressions.

.. versionadded:: 3.11
  The source files can be omitted if they are added later using
  :command:`target_sources`.

For ``SHARED`` and ``MODULE`` libraries the
:prop_tgt:`POSITION_INDEPENDENT_CODE` target
property is set to ``ON`` automatically.
A ``SHARED`` library may be marked with the :prop_tgt:`FRAMEWORK`
target property to create an macOS Framework.

.. versionadded:: 3.8
  A ``STATIC`` library may be marked with the :prop_tgt:`FRAMEWORK`
  target property to create a static Framework.

If a library does not export any symbols, it must not be declared as a
``SHARED`` library.  For example, a Windows resource DLL or a managed C++/CLI
DLL that exports no unmanaged symbols would need to be a ``MODULE`` library.
This is because CMake expects a ``SHARED`` library to always have an
associated import library on Windows.

By default the library file will be created in the build tree directory
corresponding to the source tree directory in which the command was
invoked.  See documentation of the :prop_tgt:`ARCHIVE_OUTPUT_DIRECTORY`,
:prop_tgt:`LIBRARY_OUTPUT_DIRECTORY`, and
:prop_tgt:`RUNTIME_OUTPUT_DIRECTORY` target properties to change this
location.  See documentation of the :prop_tgt:`OUTPUT_NAME` target
property to change the ``<name>`` part of the final file name.

See the :manual:`cmake-buildsystem(7)` manual for more on defining
buildsystem properties.

See also :prop_sf:`HEADER_FILE_ONLY` on what to do if some sources are
pre-processed, and you want to have the original sources reachable from
within IDE.

.. versionchanged:: 3.30

  On platforms that do not support shared libraries, ``add_library``
  now fails on calls creating ``SHARED`` libraries instead of
  automatically converting them to ``STATIC`` libraries as before.
  See policy :policy:`CMP0164`.

Object Libraries
^^^^^^^^^^^^^^^^

.. signature::
  add_library(<name> OBJECT <sources>...)
  :target: OBJECT

  Add an :ref:`Object Library <Object Libraries>` to compile source files
  without archiving or linking their object files into a library.

Other targets created by ``add_library`` or :command:`add_executable`
may reference the objects using an expression of the
form :genex:`$\<TARGET_OBJECTS:objlib\> <TARGET_OBJECTS>` as a source, where
``objlib`` is the object library name.  For example:

.. code-block:: cmake

  add_library(... $<TARGET_OBJECTS:objlib> ...)
  add_executable(... $<TARGET_OBJECTS:objlib> ...)

will include objlib's object files in a library and an executable
along with those compiled from their own sources.  Object libraries
may contain only sources that compile, header files, and other files
that would not affect linking of a normal library (e.g. ``.txt``).
They may contain custom commands generating such sources, but not
``PRE_BUILD``, ``PRE_LINK``, or ``POST_BUILD`` commands.  Some native build
systems (such as :generator:`Xcode`) may not like targets that have only
object files, so consider adding at least one real source file to any target
that references :genex:`$\<TARGET_OBJECTS:objlib\> <TARGET_OBJECTS>`.

.. versionadded:: 3.12
  Object libraries can be linked to with :command:`target_link_libraries`.

Interface Libraries
^^^^^^^^^^^^^^^^^^^

.. signature::
  add_library(<name> INTERFACE)
  :target: INTERFACE

  Add an :ref:`Interface Library <Interface Libraries>` target that may
  specify usage requirements for dependents but does not compile sources
  and does not produce a library artifact on disk.

  An interface library with no source files is not included as a target
  in the generated buildsystem.  However, it may have
  properties set on it and it may be installed and exported.
  Typically, ``INTERFACE_*`` properties are populated on an interface
  target using the commands:

  * :command:`set_property`,
  * :command:`target_link_libraries(INTERFACE)`,
  * :command:`target_link_options(INTERFACE)`,
  * :command:`target_include_directories(INTERFACE)`,
  * :command:`target_compile_options(INTERFACE)`,
  * :command:`target_compile_definitions(INTERFACE)`, and
  * :command:`target_sources(INTERFACE)`,

  and then it is used as an argument to :command:`target_link_libraries`
  like any other target.

  .. versionadded:: 3.15
    An interface library can have :prop_tgt:`PUBLIC_HEADER` and
    :prop_tgt:`PRIVATE_HEADER` properties.  The headers specified by those
    properties can be installed using the :command:`install(TARGETS)` command.

.. signature::
  add_library(<name> INTERFACE [EXCLUDE_FROM_ALL] <sources>...)
  :target: INTERFACE-with-sources

  .. versionadded:: 3.19

  Add an :ref:`Interface Library <Interface Libraries>` target with
  source files (in addition to usage requirements and properties as
  documented by the :command:`above signature <add_library(INTERFACE)>`).
  Source files may be listed directly in the ``add_library`` call
  or added later by calls to :command:`target_sources` with the
  ``PRIVATE`` or ``PUBLIC`` keywords.

  If an interface library has source files (i.e. the :prop_tgt:`SOURCES`
  target property is set), or header sets (i.e. the :prop_tgt:`HEADER_SETS`
  target property is set), it will appear in the generated buildsystem
  as a build target much like a target defined by the
  :command:`add_custom_target` command.  It does not compile any sources,
  but does contain build rules for custom commands created by the
  :command:`add_custom_command` command.

  The options are:

  ``EXCLUDE_FROM_ALL``
    Set the :prop_tgt:`EXCLUDE_FROM_ALL` target property automatically.
    See documentation of that target property for details.

  .. note::
    In most command signatures where the ``INTERFACE`` keyword appears,
    the items listed after it only become part of that target's usage
    requirements and are not part of the target's own settings.  However,
    in this signature of ``add_library``, the ``INTERFACE`` keyword refers
    to the library type only.  Sources listed after it in the ``add_library``
    call are ``PRIVATE`` to the interface library and do not appear in its
    :prop_tgt:`INTERFACE_SOURCES` target property.

.. signature::
  add_library(<name> INTERFACE SYMBOLIC)
  :target: INTERFACE-SYMBOLIC

  .. versionadded:: 4.2

  Add a symbolic :ref:`Interface Library <Interface Libraries>` target.
  Symbolic interface libraries are useful for representing optional components
  or features in a package.  They have no usage requirements, do not compile
  sources, and do not produce a library artifact on disk, but they may be
  exported and installed.  They can also be tested for existence with the
  regular :command:`if(TARGET)` subcommand.

  A symbolic interface library may be used as a linkable target to enforce the
  presence of optional components in a dependency.  For example, if a library
  ``libgui`` may or may not provide a feature ``widget``, a consumer package
  can link against ``widget`` to express that it requires this component to be
  available.  This allows :command:`find_package` calls that declare required
  components to be validated by linking against the corresponding symbolic
  targets.

  A symbolic interface library has the :prop_tgt:`SYMBOLIC` target property
  set to true.

.. _`add_library imported libraries`:

Imported Libraries
^^^^^^^^^^^^^^^^^^

.. signature::
  add_library(<name> <type> IMPORTED [GLOBAL])
  :target: IMPORTED

  Add an :ref:`IMPORTED library target <Imported Targets>` called ``<name>``.
  The target name may be referenced like any target built within the project,
  except that by default it is visible only in the directory in which it is
  created, and below.

  The ``<type>`` must be one of:

  ``STATIC``, ``SHARED``, ``MODULE``, ``UNKNOWN``
    References a library file located outside the project.  The
    :prop_tgt:`IMPORTED_LOCATION` target property (or its per-configuration
    variant :prop_tgt:`IMPORTED_LOCATION_<CONFIG>`) specifies the
    location of the main library file on disk:

    * For a ``SHARED`` library on most non-Windows platforms, the main library
      file is the ``.so`` or ``.dylib`` file used by both linkers and dynamic
      loaders.  If the referenced library file has a ``SONAME`` (or on macOS,
      has a ``LC_ID_DYLIB`` starting in ``@rpath/``), the value of that field
      should be set in the :prop_tgt:`IMPORTED_SONAME` target property.
      If the referenced library file does not have a ``SONAME``, but the
      platform supports it, then  the :prop_tgt:`IMPORTED_NO_SONAME` target
      property should be set.

    * For a ``SHARED`` library on Windows, the :prop_tgt:`IMPORTED_IMPLIB`
      target property (or its per-configuration variant
      :prop_tgt:`IMPORTED_IMPLIB_<CONFIG>`) specifies the location of the
      DLL import library file (``.lib`` or ``.dll.a``) on disk, and the
      ``IMPORTED_LOCATION`` is the location of the ``.dll`` runtime
      library (and is optional, but needed by the :genex:`TARGET_RUNTIME_DLLS`
      generator expression).

    Additional usage requirements may be specified in ``INTERFACE_*``
    properties.

    An ``UNKNOWN`` library type is typically only used in the implementation
    of :ref:`Find Modules`.  It allows the path to an imported library
    (often found using the :command:`find_library` command) to be used
    without having to know what type of library it is.  This is especially
    useful on Windows where a static library and a DLL's import library
    both have the same file extension.

  ``OBJECT``
    References a set of object files located outside the project.
    The :prop_tgt:`IMPORTED_OBJECTS` target property (or its per-configuration
    variant :prop_tgt:`IMPORTED_OBJECTS_<CONFIG>`) specifies the locations of
    object files on disk.
    Additional usage requirements may be specified in ``INTERFACE_*``
    properties.

  ``INTERFACE``
    Does not reference any library or object files on disk, but may
    specify usage requirements in ``INTERFACE_*`` properties.

  The options are:

  ``GLOBAL``
    Make the target name globally visible.

No rules are generated to build imported targets, and the :prop_tgt:`IMPORTED`
target property is ``True``.  Imported libraries are useful for convenient
reference from commands like :command:`target_link_libraries`.

Details about the imported library are specified by setting properties whose
names begin in ``IMPORTED_`` and ``INTERFACE_``.  See documentation of
such properties for more information.

Alias Libraries
^^^^^^^^^^^^^^^

.. signature::
  add_library(<name> ALIAS <target>)
  :target: ALIAS

  Creates an :ref:`Alias Target <Alias Targets>`, such that ``<name>`` can be
  used to refer to ``<target>`` in subsequent commands.  The ``<name>`` does
  not appear in the generated buildsystem as a make target.  The ``<target>``
  may not be an ``ALIAS``.

.. versionadded:: 3.11
  An ``ALIAS`` can target a ``GLOBAL`` :ref:`Imported Target <Imported Targets>`

.. versionadded:: 3.18
  An ``ALIAS`` can target a non-``GLOBAL`` Imported Target. Such alias is
  scoped to the directory in which it is created and below.
  The :prop_tgt:`ALIAS_GLOBAL` target property can be used to check if the
  alias is global or not.

``ALIAS`` targets can be used as linkable targets and as targets to
read properties from.  They can also be tested for existence with the
regular :command:`if(TARGET)` subcommand.  The ``<name>`` may not be used
to modify properties of ``<target>``, that is, it may not be used as the
operand of :command:`set_property`, :command:`set_target_properties`,
:command:`target_link_libraries` etc.  An ``ALIAS`` target may not be
installed or exported.

See Also
^^^^^^^^

* :command:`add_executable`
