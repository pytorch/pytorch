This variable controls whether the :variable:`CMAKE_FIND_ROOT_PATH` and
:variable:`CMAKE_SYSROOT` are used by |FIND_XXX|.

If set to ``ONLY``, then only the roots in :variable:`CMAKE_FIND_ROOT_PATH`
will be searched. If set to ``NEVER``, then the roots in
:variable:`CMAKE_FIND_ROOT_PATH` will be ignored and only the host system
root will be used. If set to ``BOTH``, then the host system paths and the
paths in :variable:`CMAKE_FIND_ROOT_PATH` will be searched.
