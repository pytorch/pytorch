cmake_path
----------

.. versionadded:: 3.20

This command is for the manipulation of paths.  Only syntactic aspects of
paths are handled, there is no interaction of any kind with any underlying
file system.  The path may represent a non-existing path or even one that
is not allowed to exist on the current file system or platform.
For operations that do interact with the filesystem, see the :command:`file`
command.

.. note::

  The ``cmake_path`` command handles paths in the format of the build system
  (i.e. the host platform), not the target system.  When cross-compiling,
  if the path contains elements that are not representable on the host
  platform (e.g. a drive letter when the host is not Windows), the results
  will be unpredictable.

Synopsis
^^^^^^^^

.. parsed-literal::

  `Conventions`_

  `Path Structure And Terminology`_

  `Normalization`_

  `Decomposition`_
    cmake_path(`GET`_ <path-var> `ROOT_NAME <GET ... ROOT_NAME_>`_ <out-var>)
    cmake_path(`GET`_ <path-var> `ROOT_DIRECTORY <GET ... ROOT_DIRECTORY_>`_ <out-var>)
    cmake_path(`GET`_ <path-var> `ROOT_PATH <GET ... ROOT_PATH_>`_ <out-var>)
    cmake_path(`GET`_ <path-var> `FILENAME <GET ... FILENAME_>`_ <out-var>)
    cmake_path(`GET`_ <path-var> `EXTENSION <GET ... EXTENSION_>`_ [LAST_ONLY] <out-var>)
    cmake_path(`GET`_ <path-var> `STEM <GET ... STEM_>`_ [LAST_ONLY] <out-var>)
    cmake_path(`GET`_ <path-var> `RELATIVE_PART <GET ... RELATIVE_PART_>`_ <out-var>)
    cmake_path(`GET`_ <path-var> `PARENT_PATH <GET ... PARENT_PATH_>`_ <out-var>)

  `Query`_
    cmake_path(`HAS_ROOT_NAME`_ <path-var> <out-var>)
    cmake_path(`HAS_ROOT_DIRECTORY`_ <path-var> <out-var>)
    cmake_path(`HAS_ROOT_PATH`_ <path-var> <out-var>)
    cmake_path(`HAS_FILENAME`_ <path-var> <out-var>)
    cmake_path(`HAS_EXTENSION`_ <path-var> <out-var>)
    cmake_path(`HAS_STEM`_ <path-var> <out-var>)
    cmake_path(`HAS_RELATIVE_PART`_ <path-var> <out-var>)
    cmake_path(`HAS_PARENT_PATH`_ <path-var> <out-var>)
    cmake_path(`IS_ABSOLUTE`_ <path-var> <out-var>)
    cmake_path(`IS_RELATIVE`_ <path-var> <out-var>)
    cmake_path(`IS_PREFIX`_ <path-var> <input> [NORMALIZE] <out-var>)

  `Comparison`_
    cmake_path(`COMPARE`_ <input1> <op> <input2> <out-var>)

  `Modification`_
    cmake_path(`SET`_ <path-var> [NORMALIZE] <input>)
    cmake_path(`APPEND`_ <path-var> [<input>...] [OUTPUT_VARIABLE <out-var>])
    cmake_path(`APPEND_STRING`_ <path-var> [<input>...] [OUTPUT_VARIABLE <out-var>])
    cmake_path(`REMOVE_FILENAME`_ <path-var> [OUTPUT_VARIABLE <out-var>])
    cmake_path(`REPLACE_FILENAME`_ <path-var> <input> [OUTPUT_VARIABLE <out-var>])
    cmake_path(`REMOVE_EXTENSION`_ <path-var> [LAST_ONLY] [OUTPUT_VARIABLE <out-var>])
    cmake_path(`REPLACE_EXTENSION`_ <path-var> [LAST_ONLY] <input> [OUTPUT_VARIABLE <out-var>])

  `Generation`_
    cmake_path(`NORMAL_PATH`_ <path-var> [OUTPUT_VARIABLE <out-var>])
    cmake_path(`RELATIVE_PATH`_ <path-var> [BASE_DIRECTORY <input>] [OUTPUT_VARIABLE <out-var>])
    cmake_path(`ABSOLUTE_PATH`_ <path-var> [BASE_DIRECTORY <input>] [NORMALIZE] [OUTPUT_VARIABLE <out-var>])

  `Native Conversion`_
    cmake_path(`NATIVE_PATH`_ <path-var> [NORMALIZE] <out-var>)
    cmake_path(`CONVERT`_ <input> `TO_CMAKE_PATH_LIST <CONVERT ... TO_CMAKE_PATH_LIST_>`_ <out-var> [NORMALIZE])
    cmake_path(`CONVERT`_ <input> `TO_NATIVE_PATH_LIST <CONVERT ... TO_NATIVE_PATH_LIST_>`_ <out-var> [NORMALIZE])

  `Hashing`_
    cmake_path(`HASH`_ <path-var> <out-var>)

Conventions
^^^^^^^^^^^

The following conventions are used in this command's documentation:

``<path-var>``
  Always the name of a variable.  For commands that expect a ``<path-var>``
  as input, the variable must exist and it is expected to hold a single path.

``<input>``
  A string literal which may contain a path, path fragment, or multiple paths
  with a special separator depending on the command.  See the description of
  each command to see how this is interpreted.

``<input>...``
  Zero or more string literal arguments.

``<out-var>``
  The name of a variable into which the result of a command will be written.


.. _Path Structure And Terminology:

Path Structure And Terminology
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A path has the following structure (all components are optional, with some
constraints):

::

  root-name root-directory-separator (item-name directory-separator)* filename

``root-name``
  Identifies the root on a filesystem with multiple roots (such as ``"C:"``
  or ``"//myserver"``). It is optional.

``root-directory-separator``
  A directory separator that, if present, indicates that this path is
  absolute.  If it is missing and the first element other than the
  ``root-name`` is an ``item-name``, then the path is relative.

``item-name``
  A sequence of characters that aren't directory separators.  This name may
  identify a file, a hard link, a symbolic link, or a directory.  Two special
  cases are recognized:

  * The item name consisting of a single dot character ``.`` is a
    directory name that refers to the current directory.

  * The item name consisting of two dot characters ``..`` is a
    directory name that refers to the parent directory.

  The ``(...)*`` pattern shown above is to indicate that there can be zero
  or more item names, with multiple items separated by a
  ``directory-separator``.  The ``()*`` characters are not part of the path.

``directory-separator``
  The only recognized directory separator is a forward slash character ``/``.
  If this character is repeated, it is treated as a single directory
  separator.  In other words, ``/usr///////lib`` is the same as ``/usr/lib``.

.. _FILENAME_DEF:
.. _EXTENSION_DEF:
.. _STEM_DEF:

``filename``
  A path has a ``filename`` if it does not end with a ``directory-separator``.
  The ``filename`` is effectively the last ``item-name`` of the path, so it
  can also be a hard link, symbolic link or a directory.

  A ``filename`` can have an *extension*.  By default, the extension is
  defined as the sub-string beginning at the left-most period (including
  the period) and until the end of the ``filename``.  In commands that
  accept a ``LAST_ONLY`` keyword, ``LAST_ONLY`` changes the interpretation
  to the sub-string beginning at the right-most period.

  The following exceptions apply to the above interpretation:

  * If the first character in the ``filename`` is a period, that period is
    ignored (i.e. a ``filename`` like ``".profile"`` is treated as having
    no extension).

  * If the ``filename`` is either ``.`` or ``..``, it has no extension.

  The *stem* is the part of the ``filename`` before the extension.

Some commands refer to a ``root-path``.  This is the concatenation of
``root-name`` and ``root-directory-separator``, either or both of which can
be empty.  A ``relative-part`` refers to the full path with any ``root-path``
removed.


Creating A Path Variable
^^^^^^^^^^^^^^^^^^^^^^^^

While a path can be created with care using an ordinary :command:`set`
command, it is recommended to use :command:`cmake_path(SET)` instead, as it
automatically converts the path to the required form where required.  The
:command:`cmake_path(APPEND)` subcommand may be another suitable alternative
where a path needs to be constructed by joining fragments. The following
example compares the three methods for constructing the same path:

.. code-block:: cmake

  set(path1 "${CMAKE_CURRENT_SOURCE_DIR}/data")

  cmake_path(SET path2 "${CMAKE_CURRENT_SOURCE_DIR}/data")

  cmake_path(APPEND path3 "${CMAKE_CURRENT_SOURCE_DIR}" "data")

`Modification`_ and `Generation`_ sub-commands can either store the result
in-place, or in a separate variable named after an ``OUTPUT_VARIABLE``
keyword.  All other sub-commands store the result in a mandatory ``<out-var>``
variable.

.. _Normalization:

Normalization
^^^^^^^^^^^^^

Some sub-commands support *normalizing* a path.  The algorithm used to
normalize a path is as follows:

1. If the path is empty, stop (the normalized form of an empty path is
   also an empty path).
2. Replace each ``directory-separator``, which may consist of multiple
   separators, with a single ``/`` (``/a///b  --> /a/b``).
3. Remove each solitary period (``.``) and any immediately following
   ``directory-separator`` (``/a/./b/. --> /a/b``).
4. Remove each ``item-name`` (other than ``..``) that is immediately
   followed by a ``directory-separator`` and a ``..``, along with any
   immediately following ``directory-separator`` (``/a/b/../c --> a/c``).
5. If there is a ``root-directory``, remove any ``..`` and any
   ``directory-separators`` immediately following them.  The parent of the
   root directory is treated as still the root directory (``/../a --> /a``).
6. If the last ``item-name`` is ``..``, remove any trailing
   ``directory-separator`` (``../ --> ..``).
7. If the path is empty by this stage, add a ``dot`` (normal form of ``./``
   is ``.``).


.. _Path Decomposition:

Decomposition
^^^^^^^^^^^^^

.. _GET:

The following forms of the ``GET`` subcommand each retrieve a different
component or group of components from a path.  See
`Path Structure And Terminology`_ for the meaning of each path component.

.. signature::
  cmake_path(GET <path-var> ROOT_NAME <out-var>)
  cmake_path(GET <path-var> ROOT_DIRECTORY <out-var>)
  cmake_path(GET <path-var> ROOT_PATH <out-var>)
  cmake_path(GET <path-var> FILENAME <out-var>)
  cmake_path(GET <path-var> EXTENSION [LAST_ONLY] <out-var>)
  cmake_path(GET <path-var> STEM [LAST_ONLY] <out-var>)
  cmake_path(GET <path-var> RELATIVE_PART <out-var>)
  cmake_path(GET <path-var> PARENT_PATH <out-var>)
  :target:
    GET ... ROOT_NAME
    GET ... ROOT_DIRECTORY
    GET ... ROOT_PATH
    GET ... FILENAME
    GET ... EXTENSION
    GET ... STEM
    GET ... RELATIVE_PART
    GET ... PARENT_PATH

  If a requested component is not present in the path, an empty string will be
  stored in ``<out-var>``.  For example, only Windows systems have the concept
  of a ``root-name``, so when the host machine is non-Windows, the ``ROOT_NAME``
  subcommand will always return an empty string.

  For ``PARENT_PATH``, if the :cref:`HAS_RELATIVE_PART` sub-command returns
  false, the result is a copy of ``<path-var>``.  Note that this implies that a
  root directory is considered to have a parent, with that parent being itself.
  Where :cref:`HAS_RELATIVE_PART` returns true, the result will essentially be
  ``<path-var>`` with one less element.

Root examples
"""""""""""""

.. code-block:: cmake

  set(path "c:/a")

  cmake_path(GET path ROOT_NAME rootName)
  cmake_path(GET path ROOT_DIRECTORY rootDir)
  cmake_path(GET path ROOT_PATH rootPath)

  message("Root name is \"${rootName}\"")
  message("Root directory is \"${rootDir}\"")
  message("Root path is \"${rootPath}\"")

::

  Root name is "c:"
  Root directory is "/"
  Root path is "c:/"

Filename examples
"""""""""""""""""

.. code-block:: cmake

  set(path "/a/b")
  cmake_path(GET path FILENAME filename)
  message("First filename is \"${filename}\"")

  # Trailing slash means filename is empty
  set(path "/a/b/")
  cmake_path(GET path FILENAME filename)
  message("Second filename is \"${filename}\"")

::

  First filename is "b"
  Second filename is ""

Extension and stem examples
"""""""""""""""""""""""""""

.. code-block:: cmake

  set(path "name.ext1.ext2")

  cmake_path(GET path EXTENSION fullExt)
  cmake_path(GET path STEM fullStem)
  message("Full extension is \"${fullExt}\"")
  message("Full stem is \"${fullStem}\"")

  # Effect of LAST_ONLY
  cmake_path(GET path EXTENSION LAST_ONLY lastExt)
  cmake_path(GET path STEM LAST_ONLY lastStem)
  message("Last extension is \"${lastExt}\"")
  message("Last stem is \"${lastStem}\"")

  # Special cases
  set(dotPath "/a/.")
  set(dotDotPath "/a/..")
  set(someMorePath "/a/.some.more")
  cmake_path(GET dotPath EXTENSION dotExt)
  cmake_path(GET dotPath STEM dotStem)
  cmake_path(GET dotDotPath EXTENSION dotDotExt)
  cmake_path(GET dotDotPath STEM dotDotStem)
  cmake_path(GET someMorePath EXTENSION someMoreExt)
  cmake_path(GET someMorePath STEM someMoreStem)
  message("Dot extension is \"${dotExt}\"")
  message("Dot stem is \"${dotStem}\"")
  message("Dot-dot extension is \"${dotDotExt}\"")
  message("Dot-dot stem is \"${dotDotStem}\"")
  message(".some.more extension is \"${someMoreExt}\"")
  message(".some.more stem is \"${someMoreStem}\"")

::

  Full extension is ".ext1.ext2"
  Full stem is "name"
  Last extension is ".ext2"
  Last stem is "name.ext1"
  Dot extension is ""
  Dot stem is "."
  Dot-dot extension is ""
  Dot-dot stem is ".."
  .some.more extension is ".more"
  .some.more stem is ".some"

Relative part examples
""""""""""""""""""""""

.. code-block:: cmake

  set(path "c:/a/b")
  cmake_path(GET path RELATIVE_PART result)
  message("Relative part is \"${result}\"")

  set(path "c/d")
  cmake_path(GET path RELATIVE_PART result)
  message("Relative part is \"${result}\"")

  set(path "/")
  cmake_path(GET path RELATIVE_PART result)
  message("Relative part is \"${result}\"")

::

  Relative part is "a/b"
  Relative part is "c/d"
  Relative part is ""

Path traversal examples
"""""""""""""""""""""""

.. code-block:: cmake

  set(path "c:/a/b")
  cmake_path(GET path PARENT_PATH result)
  message("Parent path is \"${result}\"")

  set(path "c:/")
  cmake_path(GET path PARENT_PATH result)
  message("Parent path is \"${result}\"")

::

  Parent path is "c:/a"
  Parent path is "c:/"


.. _Path Query:

Query
^^^^^

Each of the `cmake_path(GET) <GET_>`_ subcommands has a corresponding
``HAS_...`` subcommand which can be used to discover whether a particular path
component is present.  See `Path Structure And Terminology`_ for the
meaning of each path component.

.. signature::
  cmake_path(HAS_ROOT_NAME <path-var> <out-var>)
  cmake_path(HAS_ROOT_DIRECTORY <path-var> <out-var>)
  cmake_path(HAS_ROOT_PATH <path-var> <out-var>)
  cmake_path(HAS_FILENAME <path-var> <out-var>)
  cmake_path(HAS_EXTENSION <path-var> <out-var>)
  cmake_path(HAS_STEM <path-var> <out-var>)
  cmake_path(HAS_RELATIVE_PART <path-var> <out-var>)
  cmake_path(HAS_PARENT_PATH <path-var> <out-var>)

  Each of the above follows the predictable pattern of setting ``<out-var>``
  to true if the path has the associated component, or false otherwise.
  Note the following special cases:

  * For ``HAS_ROOT_PATH``, a true result will only be returned if at least one
    of ``root-name`` or ``root-directory`` is non-empty.

  * For ``HAS_PARENT_PATH``, the root directory is also considered to have a
    parent, which will be itself.  The result is true except if the path
    consists of just a :ref:`filename <FILENAME_DEF>`.

.. signature::
  cmake_path(IS_ABSOLUTE <path-var> <out-var>)

  Sets ``<out-var>`` to true if ``<path-var>`` is absolute.  An absolute path
  is a path that unambiguously identifies the location of a file without
  reference to an additional starting location.  On Windows, this means the
  path must have both a ``root-name`` and a ``root-directory-separator`` to be
  considered absolute.  On other platforms, just a ``root-directory-separator``
  is sufficient.  Note that this means on Windows, ``IS_ABSOLUTE`` can be
  false while :cref:`HAS_ROOT_DIRECTORY` can be true.

.. signature::
  cmake_path(IS_RELATIVE <path-var> <out-var>)

  This will store the opposite of :cref:`IS_ABSOLUTE` in ``<out-var>``.

.. signature::
  cmake_path(IS_PREFIX <path-var> <input> [NORMALIZE] <out-var>)

  Checks if ``<path-var>`` is the prefix of ``<input>``.

  When the ``NORMALIZE`` option is specified, ``<path-var>`` and ``<input>``
  are :ref:`normalized <Normalization>` before the check.

  .. code-block:: cmake

    set(path "/a/b/c")
    cmake_path(IS_PREFIX path "/a/b/c/d" result) # result = true
    cmake_path(IS_PREFIX path "/a/b" result)     # result = false
    cmake_path(IS_PREFIX path "/x/y/z" result)   # result = false

    set(path "/a/b")
    cmake_path(IS_PREFIX path "/a/c/../b" NORMALIZE result)   # result = true

.. _Path Comparison:

Comparison
^^^^^^^^^^

.. _COMPARE:

.. signature::
  cmake_path(COMPARE <input1> EQUAL <input2> <out-var>)
  cmake_path(COMPARE <input1> NOT_EQUAL <input2> <out-var>)
  :target:
    COMPARE ... EQUAL
    COMPARE ... NOT_EQUAL

  Compares the lexical representations of two paths provided as string literals.
  No normalization is performed on either path, except multiple consecutive
  directory separators are effectively collapsed into a single separator.
  Equality is determined according to the following pseudo-code logic:

  ::

    if(NOT <input1>.root_name() STREQUAL <input2>.root_name())
      return FALSE

    if(<input1>.has_root_directory() XOR <input2>.has_root_directory())
      return FALSE

    Return FALSE if a relative portion of <input1> is not lexicographically
    equal to the relative portion of <input2>. This comparison is performed path
    component-wise. If all of the components compare equal, then return TRUE.

  .. note::
    Unlike most other ``cmake_path()`` subcommands, the ``COMPARE`` subcommand
    takes literal strings as input, not the names of variables.


.. _Path Modification:

Modification
^^^^^^^^^^^^

.. signature::
  cmake_path(SET <path-var> [NORMALIZE] <input>)

  Assigns the ``<input>`` path to ``<path-var>``.  If ``<input>`` is a native
  path, it is converted into a cmake-style path with forward-slashes
  (``/``). On Windows, the long filename marker is taken into account.

  When the ``NORMALIZE`` option is specified, the path is :ref:`normalized
  <Normalization>` after the conversion.

  For example:

  .. code-block:: cmake

    set(native_path "c:\\a\\b/..\\c")
    cmake_path(SET path "${native_path}")
    message("CMake path is \"${path}\"")

    cmake_path(SET path NORMALIZE "${native_path}")
    message("Normalized CMake path is \"${path}\"")

  Output::

    CMake path is "c:/a/b/../c"
    Normalized CMake path is "c:/a/c"

.. signature::
  cmake_path(APPEND <path-var> [<input>...] [OUTPUT_VARIABLE <out-var>])

  Appends all the ``<input>`` arguments to the ``<path-var>`` using ``/`` as
  the ``directory-separator``.  Depending on the ``<input>``, the previous
  contents of ``<path-var>`` may be discarded.  For each ``<input>`` argument,
  the following algorithm (pseudo-code) applies:

  ::

    # <path> is the contents of <path-var>

    if(<input>.is_absolute() OR
      (<input>.has_root_name() AND
        NOT <input>.root_name() STREQUAL <path>.root_name()))
      replace <path> with <input>
      return()
    endif()

    if(<input>.has_root_directory())
      remove any root-directory and the entire relative path from <path>
    elseif(<path>.has_filename() OR
          (NOT <path-var>.has_root_directory() OR <path>.is_absolute()))
      append directory-separator to <path>
    endif()

    append <input> omitting any root-name to <path>

.. signature::
  cmake_path(APPEND_STRING <path-var> [<input>...] [OUTPUT_VARIABLE <out-var>])

  Appends all the ``<input>`` arguments to the ``<path-var>`` without adding any
  ``directory-separator``.

.. signature::
  cmake_path(REMOVE_FILENAME <path-var> [OUTPUT_VARIABLE <out-var>])

  Removes the :ref:`filename <FILENAME_DEF>` component (as returned by
  :cref:`GET ... FILENAME`) from ``<path-var>``.  After removal, any trailing
  ``directory-separator`` is left alone, if present.

  If ``OUTPUT_VARIABLE`` is not given, then after this function returns,
  :cref:`HAS_FILENAME` returns false for ``<path-var>``.

  For example:

  .. code-block:: cmake

    set(path "/a/b")
    cmake_path(REMOVE_FILENAME path)
    message("First path is \"${path}\"")

    # filename is now already empty, the following removes nothing
    cmake_path(REMOVE_FILENAME path)
    message("Second path is \"${path}\"")

  Output::

    First path is "/a/"
    Second path is "/a/"

.. signature::
  cmake_path(REPLACE_FILENAME <path-var> <input> [OUTPUT_VARIABLE <out-var>])

  Replaces the :ref:`filename <FILENAME_DEF>` component from ``<path-var>``
  with ``<input>``.  If ``<path-var>`` has no filename component (i.e.
  :cref:`HAS_FILENAME` returns false), the path is unchanged. The operation is
  equivalent to the following:

  .. code-block:: cmake

    cmake_path(HAS_FILENAME path has_filename)
    if(has_filename)
      cmake_path(REMOVE_FILENAME path)
      cmake_path(APPEND path "${input}")
    endif()

.. signature::
  cmake_path(REMOVE_EXTENSION <path-var> [LAST_ONLY]
                                         [OUTPUT_VARIABLE <out-var>])

  Removes the :ref:`extension <EXTENSION_DEF>`, if any, from ``<path-var>``.

.. signature::
  cmake_path(REPLACE_EXTENSION <path-var> [LAST_ONLY] <input>
                               [OUTPUT_VARIABLE <out-var>])

  Replaces the :ref:`extension <EXTENSION_DEF>` with ``<input>``.  Its effect
  is equivalent to the following:

  .. code-block:: cmake

    cmake_path(REMOVE_EXTENSION path)
    if(NOT input MATCHES "^\\.")
      cmake_path(APPEND_STRING path ".")
    endif()
    cmake_path(APPEND_STRING path "${input}")


.. _Path Generation:

Generation
^^^^^^^^^^

.. signature::
  cmake_path(NORMAL_PATH <path-var> [OUTPUT_VARIABLE <out-var>])

  Normalizes ``<path-var>`` according the steps described in
  :ref:`Normalization`.

.. signature::
  cmake_path(RELATIVE_PATH <path-var> [BASE_DIRECTORY <input>]
                                      [OUTPUT_VARIABLE <out-var>])

  Modifies ``<path-var>`` to make it relative to the ``BASE_DIRECTORY`` argument.
  If ``BASE_DIRECTORY`` is not specified, the default base directory will be
  :variable:`CMAKE_CURRENT_SOURCE_DIR`.

  For reference, the algorithm used to compute the relative path is the same
  as that used by C++
  `std::filesystem::path::lexically_relative
  <https://en.cppreference.com/w/cpp/filesystem/path/lexically_normal>`_.

.. signature::
  cmake_path(ABSOLUTE_PATH <path-var> [BASE_DIRECTORY <input>] [NORMALIZE]
                                      [OUTPUT_VARIABLE <out-var>])

  If ``<path-var>`` is a relative path (:cref:`IS_RELATIVE` is true), it is
  evaluated relative to the given base directory specified by ``BASE_DIRECTORY``
  option. If ``BASE_DIRECTORY`` is not specified, the default base directory
  will be :variable:`CMAKE_CURRENT_SOURCE_DIR`.

  When the ``NORMALIZE`` option is specified, the path is :ref:`normalized
  <Normalization>` after the path computation.

  Because ``cmake_path()`` does not access the filesystem, symbolic links are
  not resolved and any leading tilde is not expanded.  To compute a real path
  with symbolic links resolved and leading tildes expanded, use the
  :command:`file(REAL_PATH)` command instead.

Native Conversion
^^^^^^^^^^^^^^^^^

For commands in this section, *native* refers to the host platform, not the
target platform when cross-compiling.

.. signature::
  cmake_path(NATIVE_PATH <path-var> [NORMALIZE] <out-var>)

  Converts a cmake-style ``<path-var>`` into a native path with
  platform-specific slashes (``\`` on Windows hosts and ``/`` elsewhere).

  When the ``NORMALIZE`` option is specified, the path is :ref:`normalized
  <Normalization>` before the conversion.

.. _CONVERT:

.. signature::
  cmake_path(CONVERT <input> TO_CMAKE_PATH_LIST <out-var> [NORMALIZE])
  :target:
    CONVERT ... TO_CMAKE_PATH_LIST

  Converts a native ``<input>`` path into a cmake-style path with forward
  slashes (``/``).  On Windows hosts, the long filename marker is taken into
  account.  The input can be a single path or a system search path like
  ``$ENV{PATH}``.  A search path will be converted to a cmake-style list
  separated by ``;`` characters (on non-Windows platforms, this essentially
  means ``:`` separators are replaced with ``;``).  The result of the
  conversion is stored in the ``<out-var>`` variable.

  When the ``NORMALIZE`` option is specified, the path is :ref:`normalized
  <Normalization>` before the conversion.

  .. note::
    Unlike most other ``cmake_path()`` subcommands, the ``CONVERT`` subcommand
    takes a literal string as input, not the name of a variable.

.. signature::
  cmake_path(CONVERT <input> TO_NATIVE_PATH_LIST <out-var> [NORMALIZE])
  :target:
    CONVERT ... TO_NATIVE_PATH_LIST

  Converts a cmake-style ``<input>`` path into a native path with
  platform-specific slashes (``\`` on Windows hosts and ``/`` elsewhere).
  The input can be a single path or a cmake-style list.  A list will be
  converted into a native search path (``;``-separated on Windows,
  ``:``-separated on other platforms).  The result of the conversion is
  stored in the ``<out-var>`` variable.

  When the ``NORMALIZE`` option is specified, the path is :ref:`normalized
  <Normalization>` before the conversion.

  .. note::
    Unlike most other ``cmake_path()`` subcommands, the ``CONVERT`` subcommand
    takes a literal string as input, not the name of a variable.

  For example:

  .. code-block:: cmake

    set(paths "/a/b/c" "/x/y/z")
    cmake_path(CONVERT "${paths}" TO_NATIVE_PATH_LIST native_paths)
    message("Native path list is \"${native_paths}\"")

  Output on Windows::

    Native path list is "\a\b\c;\x\y\z"

  Output on all other platforms::

    Native path list is "/a/b/c:/x/y/z"

Hashing
^^^^^^^

.. signature::
  cmake_path(HASH <path-var> <out-var>)

  Computes a hash value of ``<path-var>`` such that for two paths ``p1`` and
  ``p2`` that compare equal (:cref:`COMPARE ... EQUAL`), the hash value of
  ``p1`` is equal to the hash value of ``p2``.  The path is always
  :ref:`normalized <Normalization>` before the hash is computed.
