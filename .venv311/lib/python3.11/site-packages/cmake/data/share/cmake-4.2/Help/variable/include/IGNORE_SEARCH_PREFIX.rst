:ref:`Semicolon-separated list <CMake Language Lists>` of search *prefixes*
to be ignored by the :command:`find_program`, :command:`find_library`,
:command:`find_file`, and :command:`find_path` commands.
The prefixes are also ignored by the *Config mode* of the
:command:`find_package` command (*Module mode* is unaffected).
To ignore specific directories instead, see |CMAKE_IGNORE_NONPREFIX_VAR|.
