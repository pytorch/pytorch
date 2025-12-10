FIND_LIBRARY_USE_OPENBSD_VERSIONING
-----------------------------------

Whether :command:`find_library` should find OpenBSD-style shared
libraries.

This property is a boolean specifying whether the
:command:`find_library` command should find shared libraries with
OpenBSD-style versioned extension: ".so.<major>.<minor>".  The
property is set to true on OpenBSD and false on other platforms.
