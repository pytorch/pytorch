CMAKE_NETRC
-----------

.. versionadded:: 3.11

This variable is used to initialize the ``NETRC`` option for the
:command:`file(DOWNLOAD)` and :command:`file(UPLOAD)` commands.
See those commands for additional information.

This variable is also used by the :module:`ExternalProject` and
:module:`FetchContent` modules for internal calls to :command:`file(DOWNLOAD)`.

The local option takes precedence over this variable.
