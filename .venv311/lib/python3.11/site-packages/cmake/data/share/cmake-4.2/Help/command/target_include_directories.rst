target_include_directories
--------------------------

Add include directories to a target.

.. code-block:: cmake

  target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
    <INTERFACE|PUBLIC|PRIVATE> [items1...]
    [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])

Specifies include directories to use when compiling a given target.
The named ``<target>`` must have been created by a command such
as :command:`add_executable` or :command:`add_library` and must not be an
:ref:`ALIAS target <Alias Targets>`.

By using ``AFTER`` or ``BEFORE`` explicitly, you can select between appending
and prepending, independent of the default.

The ``INTERFACE``, ``PUBLIC`` and ``PRIVATE`` keywords are required to specify
the :ref:`scope <Target Command Scope>` of the following arguments.
``PRIVATE`` and ``PUBLIC`` items will populate the :prop_tgt:`INCLUDE_DIRECTORIES`
property of ``<target>``. ``PUBLIC`` and ``INTERFACE`` items will populate the
:prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` property of ``<target>``.
The following arguments specify include directories.

.. versionadded:: 3.11
  Allow setting ``INTERFACE`` items on :ref:`IMPORTED targets <Imported Targets>`.

Repeated calls for the same ``<target>`` append items in the order called.

If ``SYSTEM`` is specified, the compiler will be told the directories
are meant as system include directories on some platforms.  This may
have effects such as suppressing warnings or skipping the contained
headers in dependency calculations (see compiler documentation).
Additionally, system include directories are searched after normal
include directories regardless of the order specified.

If ``SYSTEM`` is used together with ``PUBLIC`` or ``INTERFACE``, the
:prop_tgt:`INTERFACE_SYSTEM_INCLUDE_DIRECTORIES` target property will be
populated with the specified directories.

.. |command_name| replace:: ``target_include_directories``
.. include:: include/GENEX_NOTE.rst

Specified include directories may be absolute paths or relative paths.
A relative path will be interpreted as relative to the current source
directory (i.e. :variable:`CMAKE_CURRENT_SOURCE_DIR`) and converted to an
absolute path before storing it in the associated target property.
If the path starts with a generator expression, it will always be assumed
to be an absolute path (with one exception noted below) and will be used
unmodified.

Include directories usage requirements commonly differ between the build-tree
and the install-tree.  The :genex:`BUILD_INTERFACE` and
:genex:`INSTALL_INTERFACE` generator expressions can be used to describe
separate usage requirements based on the usage location.  Relative paths
are allowed within the :genex:`INSTALL_INTERFACE` expression and are
interpreted as relative to the installation prefix.  Relative paths should not
be used in :genex:`BUILD_INTERFACE` expressions because they will not be
converted to absolute.  For example:

.. code-block:: cmake

  target_include_directories(mylib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/mylib>
    $<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
  )

Creating Relocatable Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |INTERFACE_PROPERTY_LINK| replace:: :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES`
.. include:: /include/INTERFACE_INCLUDE_DIRECTORIES_WARNING.rst

See Also
^^^^^^^^

* :command:`include_directories`
* :command:`target_compile_definitions`
* :command:`target_compile_features`
* :command:`target_compile_options`
* :command:`target_link_libraries`
* :command:`target_link_directories`
* :command:`target_link_options`
* :command:`target_precompile_headers`
* :command:`target_sources`
