link_directories
----------------

Add directories in which the linker will look for libraries.

.. code-block:: cmake

  link_directories([AFTER|BEFORE] directory1 [directory2 ...])

Adds the paths in which the linker should search for libraries.
Relative paths given to this command are interpreted as relative to
the current source directory, see :policy:`CMP0015`.

The command will apply only to targets created after it is called.

.. versionadded:: 3.13
  The directories are added to the :prop_dir:`LINK_DIRECTORIES` directory
  property for the current ``CMakeLists.txt`` file, converting relative
  paths to absolute as needed.  See the :manual:`cmake-buildsystem(7)`
  manual for more on defining buildsystem properties.

.. versionadded:: 3.13
  By default the directories specified are appended onto the current list of
  directories.  This default behavior can be changed by setting
  :variable:`CMAKE_LINK_DIRECTORIES_BEFORE` to ``ON``.  By using
  ``AFTER`` or ``BEFORE`` explicitly, you can select between appending and
  prepending, independent of the default.

.. versionadded:: 3.13
  Arguments to ``link_directories`` may use "generator expressions" with
  the syntax "$<...>".  See the :manual:`cmake-generator-expressions(7)`
  manual for available expressions.

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

  If a library search path must be provided, prefer to localize the effect
  where possible by using the :command:`target_link_directories` command
  rather than ``link_directories()``.  The target-specific command can also
  control how the search directories propagate to other dependent targets.

See Also
^^^^^^^^

* :command:`target_link_directories`
* :command:`target_link_libraries`
