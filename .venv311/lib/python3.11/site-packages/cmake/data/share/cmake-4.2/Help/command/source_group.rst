source_group
------------

Define a grouping for source files in IDE project generation.
There are two different signatures to create source groups.

.. code-block:: cmake

  source_group(<name> [FILES <src>...] [REGULAR_EXPRESSION <regex>])
  source_group(TREE <root> [PREFIX <prefix>] [FILES <src>...])

Defines a group into which sources will be placed in project files.
This is intended to set up file tabs in Visual Studio.
The group is scoped in the directory where the command is called,
and applies to sources in targets created in that directory.

The options are:

``TREE``
 .. versionadded:: 3.8

 CMake will automatically detect, from ``<src>`` files paths, source groups
 it needs to create, to keep structure of source groups analogically to the
 actual files and directories structure in the project. Paths of ``<src>``
 files will be cut to be relative to ``<root>``. The command fails if the
 paths within ``src`` do not start with ``root``.

``PREFIX``
 .. versionadded:: 3.8

 Source group and files located directly in ``<root>`` path, will be placed
 in ``<prefix>`` source groups.

``FILES``
 Any source file specified explicitly will be placed in group
 ``<name>``.  Relative paths are interpreted with respect to the
 current source directory.

``REGULAR_EXPRESSION``
 Any source file whose name matches the regular expression will
 be placed in group ``<name>``.

If a source file matches multiple groups, the *last* group that
explicitly lists the file with ``FILES`` will be favored, if any.
If no group explicitly lists the file, the *last* group whose
regular expression matches the file will be favored.

The ``<name>`` of the group and ``<prefix>`` argument may contain forward
slashes or backslashes to specify subgroups.  Backslashes need to be escaped
appropriately:

.. code-block:: cmake

  source_group(base/subdir ...)
  source_group(outer\\inner ...)
  source_group(TREE <root> PREFIX sources\\inc ...)

.. versionadded:: 3.18
  Allow using forward slashes (``/``) to specify subgroups.

For backwards compatibility, the short-hand signature

.. code-block:: cmake

  source_group(<name> <regex>)

is equivalent to

.. code-block:: cmake

  source_group(<name> REGULAR_EXPRESSION <regex>)
