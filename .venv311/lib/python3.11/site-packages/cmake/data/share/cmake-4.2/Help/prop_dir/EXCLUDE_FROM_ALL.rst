EXCLUDE_FROM_ALL
----------------

Set this directory property to a true value on a subdirectory to exclude
its targets from the "all" target of its ancestors.  If excluded, running
e.g. ``make`` in the parent directory will not build targets in the
subdirectory by default.  This does not affect the "all" target of the
subdirectory itself.  Running e.g. ``make`` inside the subdirectory will
still build its targets.

``EXCLUDE_FROM_ALL`` is meant for when the subdirectory contains
a separate part of the project that is useful, but not necessary,
such as a set of examples, or e.g. an integrated 3rd party library.
Typically the subdirectory should contain its own :command:`project`
command invocation so that a full build system will be generated in the
subdirectory (such as a Visual Studio IDE solution file).  Note that
inter-target dependencies supersede this exclusion.  If a target built by
the parent project depends on a target in the subdirectory, the dependee
target will be included in the parent project build system to satisfy
the dependency.

If the ``EXCLUDE_FROM_ALL`` argument is provided, it has the following effects:

* Targets defined in the subdirectory or below will not be
  included in the ``ALL`` target of the parent directory.
  Those targets must be built explicitly by the user,
  or be a dependency of another target that will be built.
* Targets defined in the subdirectory or below will be
  excluded from IDE project files.
* Any install rules defined in the subdirectory or below will
  be ignored when installing the parent directory.

Note that these effects are not the same as those for the
:prop_tgt:`EXCLUDE_FROM_ALL` target property.
