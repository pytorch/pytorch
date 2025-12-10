LINK_SEARCH_END_STATIC
----------------------

End a link line such that static system libraries are used.

Some linkers support switches such as ``-Bstatic`` and ``-Bdynamic`` to
determine whether to use static or shared libraries for ``-lXXX`` options.
CMake uses these options to set the link type for libraries whose full
paths are not known or (in some cases) are in implicit link
directories for the platform.  By default CMake adds an option at the
end of the library list (if necessary) to set the linker search type
back to its starting type.  This property switches the final linker
search type to ``-Bstatic`` regardless of how it started.

This property is initialized by the value of the variable
:variable:`CMAKE_LINK_SEARCH_END_STATIC` if it is set
when a target is created.

See also :prop_tgt:`LINK_SEARCH_START_STATIC`.
