Unix Makefiles
--------------

Generates standard UNIX makefiles.

A hierarchy of UNIX makefiles is generated into the build tree.  Use
any standard UNIX-style make program to build the project through
the ``all`` target and install the project through the ``install``
(or ``install/strip``) target.

For each subdirectory ``sub/dir`` of the project a UNIX makefile will
be created, containing the following targets:

``all``
  Depends on all targets required by the subdirectory.

``install``
  Runs the install step in the subdirectory, if any.

``install/strip``
  Runs the install step in the subdirectory followed by a ``CMAKE_STRIP`` command,
  if any.

  The ``CMAKE_STRIP`` variable will contain the platform's ``strip`` utility, which
  removes symbols information from generated binaries.

``test``
  Runs the test step in the subdirectory, if any.

``package``
  Runs the package step in the subdirectory, if any.
