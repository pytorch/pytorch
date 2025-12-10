NO_SONAME
---------

Whether to set ``soname`` when linking a shared library.

Enable this boolean property if a generated ``SHARED`` library
should not have ``soname`` set.  Default is to set ``soname`` on all
shared libraries as long as the platform supports it.
Generally, use this property only for leaf private libraries or
plugins.  If you use it on normal shared libraries which other targets
link against, on some platforms a linker will insert a full path to
the library (as specified at link time) into the dynamic section of
the dependent binary.  Therefore, once installed, dynamic loader may
eventually fail to locate the library for the binary.
