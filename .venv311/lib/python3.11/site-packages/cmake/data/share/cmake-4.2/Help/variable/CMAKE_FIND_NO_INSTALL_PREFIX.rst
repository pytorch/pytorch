CMAKE_FIND_NO_INSTALL_PREFIX
----------------------------

Exclude the values of the :variable:`CMAKE_INSTALL_PREFIX` and
:variable:`CMAKE_STAGING_PREFIX` variables from
:variable:`CMAKE_SYSTEM_PREFIX_PATH`.  CMake adds these project-destination
prefixes to :variable:`CMAKE_SYSTEM_PREFIX_PATH` by default in order to
support building a series of dependent packages and installing them into
a common prefix.  Set ``CMAKE_FIND_NO_INSTALL_PREFIX`` to ``TRUE``
to suppress this behavior.

The :variable:`CMAKE_SYSTEM_PREFIX_PATH` is initialized on the first call to a
:command:`project` or :command:`enable_language` command.  Therefore one must
set ``CMAKE_FIND_NO_INSTALL_PREFIX`` before this in order to take effect.  A
user may set the variable as a cache entry on the command line to achieve this.

Note that the prefix(es) may still be searched for other reasons, such as being
the same prefix as the CMake installation, or for being a built-in system
prefix.
