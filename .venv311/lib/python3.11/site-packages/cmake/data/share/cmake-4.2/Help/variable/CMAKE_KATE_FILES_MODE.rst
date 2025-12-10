CMAKE_KATE_FILES_MODE
---------------------

.. versionadded:: 3.27

This cache variable is used by the Kate project generator and controls
to what mode the ``files`` entry in the project file will be set.  See
:manual:`cmake-generators(7)`.

Possible values are ``AUTO``, ``SVN``, ``GIT``, ``HG``, ``FOSSIL`` and ``LIST``.

When set to ``LIST``, CMake will put the list of source files known to CMake
in the project file.
When set to ``SVN``, ``GIT``, ``HG`` or ``FOSSIL``, CMake will set
the generated project accordingly to Subversion, git, Mercurial
or Fossil, and Kate will then use the respective command line tool to
retrieve the list of files in the project.
When unset or set to ``AUTO``, CMake will try to detect whether the
source directory is part of a git or svn checkout or not, and put the
respective entry into the project file.
