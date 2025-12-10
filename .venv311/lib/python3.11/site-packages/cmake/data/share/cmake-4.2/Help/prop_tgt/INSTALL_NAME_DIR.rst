INSTALL_NAME_DIR
----------------

Directory name for installed targets on Apple platforms.

``INSTALL_NAME_DIR`` is a string specifying the directory portion of the
"install_name" field of shared libraries on Apple platforms for
installed targets.  When not set, the default directory used is determined
by :prop_tgt:`MACOSX_RPATH`.  If the :prop_tgt:`BUILD_WITH_INSTALL_NAME_DIR`
property is set, this will be used already in the build tree.
Policies :policy:`CMP0068` and :policy:`CMP0042` are also relevant.

This property is initialized by the value of the variable
:variable:`CMAKE_INSTALL_NAME_DIR` if it is set when a target is
created.

This property supports :manual:`generator expressions <cmake-generator-expressions(7)>`.
In particular, the :genex:`$<INSTALL_PREFIX>` generator expression can be
used to set the directory relative to the install-time prefix.

On platforms that support runtime paths (``RPATH``), refer to the
:prop_tgt:`INSTALL_RPATH` target property.
Under Windows, the :genex:`TARGET_RUNTIME_DLLS` generator expression is related.
