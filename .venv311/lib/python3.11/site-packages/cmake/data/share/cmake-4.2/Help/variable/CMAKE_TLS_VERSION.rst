CMAKE_TLS_VERSION
-----------------

.. versionadded:: 3.30

Specify the default value for the :command:`file(DOWNLOAD)` and
:command:`file(UPLOAD)` commands' ``TLS_VERSION`` option.
If this variable is not set, the commands check the
:envvar:`CMAKE_TLS_VERSION` environment variable.
If neither is set, the default is TLS 1.2.

.. versionchanged:: 3.31
  The default is TLS 1.2.
  Previously, no minimum version was enforced by default.

The value may be one of:

.. include:: include/CMAKE_TLS_VERSION-VALUES.rst

This variable is also used by the :module:`ExternalProject` and
:module:`FetchContent` modules for internal calls to
:command:`file(DOWNLOAD)` and ``git clone``.
