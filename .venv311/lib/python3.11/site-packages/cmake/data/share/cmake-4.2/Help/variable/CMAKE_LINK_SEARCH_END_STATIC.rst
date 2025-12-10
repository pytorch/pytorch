CMAKE_LINK_SEARCH_END_STATIC
----------------------------

.. versionadded:: 3.4

End a link line such that static system libraries are used.

Some linkers support switches such as ``-Bstatic`` and ``-Bdynamic`` to
determine whether to use static or shared libraries for ``-lXXX`` options.
CMake uses these options to set the link type for libraries whose full
paths are not known or (in some cases) are in implicit link
directories for the platform.  By default CMake adds an option at the
end of the library list (if necessary) to set the linker search type
back to its starting type.  This property switches the final linker
search type to ``-Bstatic`` regardless of how it started.

This variable is used to initialize the target property
:prop_tgt:`LINK_SEARCH_END_STATIC` for all targets. If set, its
value is also used by the :command:`try_compile` command.

See also :variable:`CMAKE_LINK_SEARCH_START_STATIC`.
