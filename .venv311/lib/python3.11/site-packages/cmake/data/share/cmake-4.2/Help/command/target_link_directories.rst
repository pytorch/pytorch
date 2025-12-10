target_link_directories
-----------------------

.. versionadded:: 3.13

Add link directories to a target.

.. code-block:: cmake

  target_link_directories(<target> [BEFORE]
    <INTERFACE|PUBLIC|PRIVATE> [items1...]
    [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])

Specifies the paths in which the linker should search for libraries when
linking a given target.  Each item can be an absolute or relative path,
with the latter being interpreted as relative to the current source
directory.  These items will be added to the link command.

The named ``<target>`` must have been created by a command such as
:command:`add_executable` or :command:`add_library` and must not be an
:ref:`ALIAS target <Alias Targets>`.

The ``INTERFACE``, ``PUBLIC`` and ``PRIVATE`` keywords are required to
specify the :ref:`scope <Target Command Scope>` of the items that follow
them. ``PRIVATE`` and ``PUBLIC`` items will populate the
:prop_tgt:`LINK_DIRECTORIES` property of ``<target>``.  ``PUBLIC`` and
``INTERFACE`` items will populate the :prop_tgt:`INTERFACE_LINK_DIRECTORIES`
property of ``<target>`` (:ref:`IMPORTED targets <Imported Targets>` only
support ``INTERFACE`` items).
Each item specifies a link directory and will be converted to an absolute
path if necessary before adding it to the relevant property.  Repeated
calls for the same ``<target>`` append items in the order called.

If ``BEFORE`` is specified, the content will be prepended to the relevant
property instead of being appended.

.. |command_name| replace:: ``target_link_directories``
.. include:: include/GENEX_NOTE.rst

.. note::

  This command is rarely necessary and should be avoided where there are
  other choices.  Prefer to pass full absolute paths to libraries where
  possible, since this ensures the correct library will always be linked.
  The :command:`find_library` command provides the full path, which can
  generally be used directly in calls to :command:`target_link_libraries`.
  Situations where a library search path may be needed include:

  - Project generators like :generator:`Xcode` where the user can switch
    target architecture at build time, but a full path to a library cannot
    be used because it only provides one architecture (i.e. it is not
    a universal binary).
  - Libraries may themselves have other private library dependencies
    that expect to be found via ``RPATH`` mechanisms, but some linkers
    are not able to fully decode those paths (e.g. due to the presence
    of things like ``$ORIGIN``).

See Also
^^^^^^^^

* :command:`link_directories`
* :command:`target_compile_definitions`
* :command:`target_compile_features`
* :command:`target_compile_options`
* :command:`target_include_directories`
* :command:`target_link_libraries`
* :command:`target_link_options`
* :command:`target_precompile_headers`
* :command:`target_sources`
