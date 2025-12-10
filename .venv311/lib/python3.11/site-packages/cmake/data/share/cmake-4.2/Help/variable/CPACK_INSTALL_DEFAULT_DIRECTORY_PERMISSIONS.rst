CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
-------------------------------------------

.. versionadded:: 3.11

Default permissions for implicitly created directories during packaging.

This variable serves the same purpose during packaging as the
:variable:`CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS` variable
serves during installation (e.g. ``make install``).

If ``include(CPack)`` is used then by default this variable is set to the content
of :variable:`CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS`.
