Feature names are case-sensitive and may only contain letters, numbers
and underscores.  Feature names defined in all uppercase are reserved for
CMake's own built-in features (see `Predefined Features`_ further below).

Some aspects of feature behavior can be defined by the
:variable:`CMAKE_<LANG>_LINK_LIBRARY_<FEATURE>_ATTRIBUTES` and
:variable:`CMAKE_LINK_LIBRARY_<FEATURE>_ATTRIBUTES` variables.

Feature Definitions
^^^^^^^^^^^^^^^^^^^

A library feature definition is a list that contains one or three elements:

::

  [<PREFIX>] <LIBRARY_EXPRESSION> [<SUFFIX>]

When ``<PREFIX>`` and ``<SUFFIX>`` are specified, they precede and follow
respectively the whole list of libraries specified in the
:genex:`LINK_LIBRARY` expression, not each library item individually.
There is no guarantee that the list of specified libraries will be kept
grouped together though, so the ``<PREFIX>`` and ``<SUFFIX>`` may appear
more than once if the library list is reorganized by CMake to satisfy other
constraints.  This means constructs like ``--start-group`` and ``--end-group``,
as supported by the GNU ``ld`` linker, cannot be used in this way.  The
:genex:`LINK_GROUP` generator expression should be used instead for such
constructs.

``<LIBRARY_EXPRESSION>`` is used to specify the pattern for constructing the
corresponding fragment on the linker command line for each library.
The following placeholders can be used in the expression:

* ``<LIBRARY>`` is expanded to the full path to the library for CMake targets,
  or to a platform-specific value based on the item otherwise (the same as
  ``<LINK_ITEM>`` on Windows, or the library base name for other platforms).
* ``<LINK_ITEM>`` is expanded to how the library would normally be linked on
  the linker command line.
* ``<LIB_ITEM>`` is expanded to the full path to the library for CMake targets,
  or the item itself exactly as specified in the ``<LIBRARY_EXPRESSION>``
  otherwise.

In addition to the above, it is possible to have one pattern for paths
(CMake targets and external libraries specified with file paths) and another
for other items specified by name only.  The ``PATH{}`` and ``NAME{}`` wrappers
can be used to provide the expansion for those two cases, respectively.
When wrappers are used, both must be present.  For example:

.. code-block:: cmake

  set(CMAKE_LINK_LIBRARY_USING_weak_library
      "PATH{-weak_library <LIBRARY>}NAME{LINKER:-weak-l<LIB_ITEM>}"
  )

For all three elements of this variable (``<PREFIX>``, ``<LIBRARY_EXPRESSION>``,
and ``<SUFFIX>``), the ``LINKER:`` prefix can be used.

.. include:: ../command/include/LINK_OPTIONS_LINKER.rst
  :start-line: 3

Examples
^^^^^^^^

Loading a whole static library
""""""""""""""""""""""""""""""

A common need is to prevent the linker from discarding any symbols from a
static library.  Different linkers use different syntax for achieving this.
The following example shows how this may be implemented for some linkers.
Note that this is for illustration purposes only.  Projects should use the
built-in ``WHOLE_ARCHIVE`` feature instead (see `Predefined Features`_), which
provides a more complete and more robust implementation of this functionality.

.. code-block:: cmake

  set(CMAKE_C_LINK_LIBRARY_USING_load_archive_SUPPORTED TRUE)
  if(CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_C_LINK_LIBRARY_USING_load_archive "-force_load <LIB_ITEM>")
  elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_C_LINK_LIBRARY_USING_load_archive
      "LINKER:--push-state,--whole-archive"
      "<LINK_ITEM>"
      "LINKER:--pop-state"
    )
  elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_C_LINK_LIBRARY_USING_load_archive "/WHOLEARCHIVE:<LIBRARY>")
  else()
    # feature not yet supported for the other environments
    set(CMAKE_C_LINK_LIBRARY_USING_load_archive_SUPPORTED FALSE)
  endif()

  add_library(lib1 STATIC ...)
  add_library(lib2 SHARED ...)

  if(CMAKE_C_LINK_LIBRARY_USING_load_archive_SUPPORTED)
    # The -force_load Apple linker option requires a file name
    set(external_lib
      "$<IF:$<LINK_LANG_AND_ID:C,AppleClang>,libexternal.a,external>"
    )
    target_link_libraries(lib2 PRIVATE
      "$<LINK_LIBRARY:load_archive,lib1,${external_lib}>"
    )
  else()
    target_link_libraries(lib2 PRIVATE lib1 external)
  endif()

CMake will generate the following link expressions:

* ``AppleClang``: ``-force_load /path/to/lib1.a -force_load libexternal.a``
* ``GNU``: ``-Wl,--push-state,--whole-archive /path/to/lib1.a -lexternal -Wl,--pop-state``
* ``MSVC``: ``/WHOLEARCHIVE:/path/to/lib1.lib /WHOLEARCHIVE:external.lib``

Linking a library as weak
"""""""""""""""""""""""""

On macOS, it is possible to link a library in weak mode (the library and all
references are marked as weak imports).  Different flags must be used for a
library specified by file path compared to one specified by name.
This constraint can be solved using ``PATH{}`` and ``NAME{}`` wrappers.
Again, the following example shows how this may be implemented for some
linkers, but it is for illustration purposes only.  Projects should use the
built-in ``WEAK_FRAMEWORK`` or ``WEAK_LIBRARY`` features instead (see
`Predefined Features`_), which provide more complete and more robust
implementations of this functionality.

.. code-block:: cmake

  if (CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_LINK_LIBRARY_USING_weak_library
        "PATH{-weak_library <LIBRARY>}NAME{LINKER:-weak-l<LIB_ITEM>}"
    )
    set(CMAKE_LINK_LIBRARY_USING_weak_library_SUPPORTED TRUE)
  endif()

  add_library(lib SHARED ...)
  add_executable(main ...)
  if(CMAKE_LINK_LIBRARY_USING_weak_library_SUPPORTED)
    target_link_libraries(main PRIVATE "$<LINK_LIBRARY:weak_library,lib,external>")
  else()
    target_link_libraries(main PRIVATE lib external)
  endif()

CMake will generate the following linker command line fragment when linking
``main`` using the ``AppleClang`` toolchain:

``-weak_library /path/to/lib -Xlinker -weak-lexternal``.


Predefined Features
^^^^^^^^^^^^^^^^^^^

The following built-in library features are pre-defined by CMake:

.. include:: include/LINK_LIBRARY_PREDEFINED_FEATURES.rst
