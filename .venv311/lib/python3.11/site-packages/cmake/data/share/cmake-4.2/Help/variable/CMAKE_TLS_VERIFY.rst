CMAKE_TLS_VERIFY
----------------

Specify the default value for the :command:`file(DOWNLOAD)` and
:command:`file(UPLOAD)` commands' ``TLS_VERIFY`` options.
If this variable is not set, the commands check the
:envvar:`CMAKE_TLS_VERIFY` environment variable.
If neither is set, the default is *on*.

.. versionchanged:: 3.31
  The default is on.  Previously, the default was off.
  Users may set the :envvar:`CMAKE_TLS_VERIFY` environment
  variable to ``0`` to restore the old default.

This variable is also used by the :module:`ExternalProject` and
:module:`FetchContent` modules for internal calls to :command:`file(DOWNLOAD)`.

TLS verification can help provide confidence that one is connecting
to the desired server.  When downloading known content, one should
also use file hashes to verify it.

.. code-block:: cmake

  set(CMAKE_TLS_VERIFY TRUE)
