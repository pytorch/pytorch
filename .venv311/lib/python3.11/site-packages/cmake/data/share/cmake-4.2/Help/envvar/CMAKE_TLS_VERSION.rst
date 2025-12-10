CMAKE_TLS_VERSION
-----------------

.. versionadded:: 3.30

.. include:: include/ENV_VAR.rst

Specify the default value for the :command:`file(DOWNLOAD)` and
:command:`file(UPLOAD)` commands' ``TLS_VERSION`` option.
This environment variable is used if the option is not given
and the :variable:`CMAKE_TLS_VERSION` cmake variable is not set.
See that variable for allowed values.

This variable is also used by the :module:`ExternalProject` and
:module:`FetchContent` modules for internal calls to
:command:`file(DOWNLOAD)` and ``git clone``.
