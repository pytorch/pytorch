CMAKE_SYSTEM_APPBUNDLE_PATH
---------------------------

.. versionadded:: 3.4

Search path for macOS application bundles used by the :command:`find_program`,
and :command:`find_package` commands.  By default it contains the standard
directories for the current system.  It is *not* intended to be modified by
the project, use :variable:`CMAKE_APPBUNDLE_PATH` for this.
