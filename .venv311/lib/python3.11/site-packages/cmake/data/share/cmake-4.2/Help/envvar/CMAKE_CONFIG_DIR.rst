CMAKE_CONFIG_DIR
----------------

.. versionadded:: 3.31

.. include:: include/ENV_VAR.rst

Specify a CMake user-wide configuration directory for
:manual:`cmake-file-api(7)` queries.

If this environment variable is not set, the default user-wide
configuration directory is platform-specific:

- Windows: ``%LOCALAPPDATA%\CMake``
- macOS: ``$XDG_CONFIG_HOME/CMake`` if set, otherwise
  ``$HOME/Library/Application Support/CMake``
- Linux/Other: ``$XDG_CONFIG_HOME/cmake`` if set, otherwise
  ``$HOME/.config/cmake``
