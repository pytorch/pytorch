CMAKE_FASTBUILD_CAPTURE_SYSTEM_ENV
----------------------------------

.. versionadded:: 4.2

Controls capturing of the system environment into ``fbuild.bff``.
Setting it to ``OFF`` makes the invocation of all tools (compilers and other external processes) hermetic.

.. note::

   Setting this variable to ``OFF`` can break MSVC toolchains that rely on
   environment variables such as ``INCLUDE`` or ``LIB`` unless these are
   manually configured elsewhere.

Defaults to ``ON``.
