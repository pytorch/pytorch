BUILD_RPATH_USE_ORIGIN
----------------------

.. versionadded:: 3.14

Whether to use relative paths for the build ``RPATH``.

This property is initialized by the value of the variable
:variable:`CMAKE_BUILD_RPATH_USE_ORIGIN`.

On platforms that support runtime paths (``RPATH``) with the
``$ORIGIN`` token, setting this property to ``TRUE`` enables relative
paths in the build ``RPATH`` for executables and shared libraries that
point to shared libraries in the same build tree.

Normally the build ``RPATH`` of a binary contains absolute paths
to the directory of each shared library it links to.  The ``RPATH``
entries for directories contained within the build tree can be made
relative to enable relocatable builds and to help achieve reproducible
builds by omitting the build directory from the build environment.

This property has no effect on platforms that do not support the
``$ORIGIN`` token in ``RPATH``, or when the :variable:`CMAKE_SKIP_RPATH`
variable is set. The runtime path set through the
:prop_tgt:`BUILD_RPATH` target property is also unaffected by this
property.
