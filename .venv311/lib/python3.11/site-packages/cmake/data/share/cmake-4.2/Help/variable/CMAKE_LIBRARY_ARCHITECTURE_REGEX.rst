CMAKE_LIBRARY_ARCHITECTURE_REGEX
--------------------------------

Regex matching possible target architecture library directory names.

This is used to detect :variable:`CMAKE_<LANG>_LIBRARY_ARCHITECTURE` from the
implicit linker search path by matching the ``<arch>`` name.
