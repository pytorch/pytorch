:ref:`Semicolon-separated list <CMake Language Lists>` of directories
to be ignored by the various ``find...()`` commands.

For :command:`find_program`, :command:`find_library`, :command:`find_file`,
and :command:`find_path`, any file found in one of the listed directories
will be ignored. The listed directories do not apply recursively, so any
subdirectories to be ignored must also be explicitly listed.
|CMAKE_IGNORE_VAR| does not affect the search *prefixes* used by these
four commands. To ignore individual paths under a search prefix
(e.g. ``bin``, ``include``, ``lib``, etc.), each path must be listed in
|CMAKE_IGNORE_VAR| as a full absolute path. |CMAKE_IGNORE_PREFIX_VAR|
provides a more appropriate way to ignore a whole search prefix.

:command:`find_package` is also affected by |CMAKE_IGNORE_VAR|, but only
for *Config mode* searches. Any ``<Name>Config.cmake`` or
``<name>-config.cmake`` file found in one of the specified directories
will be ignored. In addition, any search *prefix* found in |CMAKE_IGNORE_VAR|
will be skipped for backward compatibility reasons, but new code should
prefer to use |CMAKE_IGNORE_PREFIX_VAR| to ignore prefixes instead.
