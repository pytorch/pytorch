CMAKE_PKG_CONFIG_DISABLE_UNINSTALLED
------------------------------------

.. versionadded:: 4.0

Enable / Disable the default "uninstalled" search behavior of the
:command:`cmake_pkg_config` command. When this variable is false, package files
with an "-uninstalled" suffix have higher priority than exact package name
matches.
